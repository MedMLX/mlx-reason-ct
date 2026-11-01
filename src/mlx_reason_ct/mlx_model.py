"""Native Metal Primus, multimodal projection and Qwen3.5 hybrid decoding.

All learned arithmetic runs on Metal. The source_bfloat16 profile preserves
NVIDIA's BF16 cast/cache boundaries; bfloat16 stores BF16 weights with FP32
accumulation and decoding.
Public bfloat16 selects source_bfloat16; public bfloat16_fp32 selects bfloat16.
Source layouts remain visible in the converted cache.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, cast

from mlx_reason_ct._payloads import JsonValue

if TYPE_CHECKING:
    from mlx_reason_ct import gated_delta, source_arithmetic
    from mlx_reason_ct._native_types import Array, Dtype, Mlx, NeuralRuntime

import numpy as np

from mlx_reason_ct.chunk_delta import chunk_delta
from mlx_reason_ct.errors import InvalidInputError
from mlx_reason_ct.generation_state import cache_capacity
from mlx_reason_ct.mlx_weights import SHARDS
from mlx_reason_ct.runtime import import_mlx

mx: Mlx = import_mlx()
nn: NeuralRuntime = cast("NeuralRuntime", import_module("mlx.nn"))
# Admit Metal before loading the adapted recurrence kernels.
if not TYPE_CHECKING:
    gated_delta = import_module("mlx_reason_ct.gated_delta")
    source_arithmetic = import_module("mlx_reason_ct.source_arithmetic")


@dataclass
class DecoderCache:
    """One request's full-attention KV and linear-attention convolution/state.

    ``layers`` holds one dict per decoder layer. Full-attention layers store
    ``k`` and ``v`` of shape ``(B, 4, tokens, 256)``. Linear-attention layers store
    ``conv`` of shape ``(B, 3, 8192)`` (the last three pre-convolution inputs) and
    ``recurrent`` of shape ``(B, 32, 128, 128)`` in (V, K) order. ``offset`` is the
    number of tokens already decoded. With ``kv_chunk_size > 0``, the KV token
    axis is allocated capacity; only ``:offset`` is valid. Attention never sees
    the unused tail. Capacity grows by chunk multiples, without evicting tokens.
    The recurrent state is stored in BF16 for ``source_bfloat16`` and FP32 for
    the other native precision profiles; recurrence arithmetic starts in FP32.
    """

    layers: list[dict[str, Array]] = field(default_factory=lambda: [{} for _ in range(32)])
    offset: int = 0
    kv_chunk_size: int = 0

    def clone(self) -> DecoderCache:
        """Snapshot array handles before continuation; updates use independent handles.

        MLX shares the evaluated backing buffers until an update needs a copy.
        A distinct array object per entry prevents Python indexed assignment from
        replacing the prefix snapshot's array handles.
        """
        return DecoderCache(
            [{name: mx.array(value) for name, value in layer.items()} for layer in self.layers],
            self.offset,
            self.kv_chunk_size,
        )

    @property
    def nbytes(self) -> int:
        """Total bytes held by all cached arrays."""
        return sum(int(array.nbytes) for layer in self.layers for array in layer.values())

    def arrays(self) -> list[Array]:
        """All cached arrays, for evaluation or size accounting."""
        return [array for layer in self.layers for array in layer.values()]


class NativeModel:
    """Pinned 3D network with bounded attention and feed-forward workspaces.

    ``precision`` is a native profile name (``float32``, ``bfloat16`` or
    ``source_bfloat16``), not the public API name; see the module docstring.
    Weights are loaded from the converted FP32 shards and cast to the profile dtype.
    """

    def __init__(self, model_dir: Path, *, precision: str = "float32") -> None:
        """Load the configs and FP32 shards, then cast weights to the profile dtype.

        Rejects unknown native profile names and cache tensors that are not FP32.
        """
        if precision not in {"float32", "bfloat16", "source_bfloat16"}:
            raise InvalidInputError("Unsupported NV-Reason-CT precision profile")
        self.dtype: Dtype = mx.float32 if precision == "float32" else mx.bfloat16
        self.precision: str = precision
        self.accumulate_float32: bool = precision == "bfloat16"
        self.config: dict[str, JsonValue] = json.loads((model_dir / "config.json").read_text())
        self.weights: dict[str, Array] = {}
        for name in SHARDS:
            self.weights.update(cast("dict[str, Array]", mx.load(str(model_dir / "mlx" / name))))
        for name, value in self.weights.items():
            if value.dtype != mx.float32:
                raise InvalidInputError(f"Native cache must store FP32 tensors: {name}")
            self.weights[name] = value.astype(self.dtype)
        # Materialize the lazily loaded and cast weights once, up front.
        mx.eval(*self.weights.values())
        self.delta = gated_delta
        self.eos_ids: tuple[int, ...] = tuple(
            json.loads((model_dir / "generation_config.json").read_text())["eos_token_id"]
        )
        # Primus uses 12 frequencies on each of three axes, interleaved pairs.
        coords = np.indices((24, 24, 24), dtype=np.float32).reshape(3, -1).T
        inv = 1 / np.power(np.float32(10000), np.arange(12, dtype=np.float32) / 12)
        phase = (coords[..., None] * inv).reshape(13824, 36).repeat(2, axis=-1)
        # Primus builds FP32 rotary constants even with BF16 learned weights.
        self.vision_sin: Array = mx.array(np.sin(phase))
        self.vision_cos: Array = mx.array(np.cos(phase))
        if self.precision == "source_bfloat16":
            self.vision_sin, self.vision_cos = source_arithmetic.vision_constants()
        # The source constructs this non-learned buffer on the host before transfer.
        # Metal pow differs by one ULP at six frequencies, magnified by positions.
        self.text_inv_freq: Array = mx.array(
            1 / np.power(np.float32(10000000), np.arange(0, 64, 2, dtype=np.float32) / 64)
        )

    @staticmethod
    def silu(x: Array) -> Array:
        """Apply SiLU in FP32 with ATen's direct division, then cast to the input dtype."""
        if x.dtype == mx.bfloat16:
            return source_arithmetic.silu(x)
        # ATen's SiLU divides directly; x * sigmoid(x) rounds twice in FP32.
        fp = x.astype(mx.float32)
        return (fp / (1 + mx.exp(-fp))).astype(x.dtype)

    @staticmethod
    def sigmoid(x: Array) -> Array:
        """Apply sigmoid with FP32 opmath and one cast back to the input dtype."""
        if x.dtype == mx.bfloat16:
            return source_arithmetic.sigmoid(x)
        # ATen computes BF16 activations in opmath FP32, then casts once.
        return (1 / (1 + mx.exp(-x.astype(mx.float32)))).astype(x.dtype)

    @staticmethod
    def gelu(x: Array) -> Array:
        """Apply the exact erf GELU in FP32, then cast back to the input dtype."""
        return nn.gelu(x.astype(mx.float32)).astype(x.dtype)

    def linear(self, x: Array, name: str, *, source_length: int | None = None) -> Array:
        """Apply the named linear layer (``x @ W.T`` plus optional bias) to ``(B, T, in)``.

        ``source_length`` is the row count of the unchunked activation. Callers that
        process ``x`` in row chunks pass it so the source-BF16 projection partitioning,
        which depends on the full row count, is chosen as in the unchunked source run.
        """
        # BF16 vision operands accumulate into FP32 without an output downcast.
        # Decoder inputs stay FP32 to preserve cached/uncached consistency.
        if self.accumulate_float32 and name.startswith("model.vision3d."):
            x = x.astype(mx.bfloat16).astype(mx.float32)
        weight = self.weights[name + ".weight"]
        bias = self.weights.get(name + ".bias")
        rows = x.shape[-2] if source_length is None else source_length
        if self.precision == "source_bfloat16":
            if (
                x.shape[:-1] == (1, 1)
                and rows == 1
                and tuple(reversed(weight.shape)) in source_arithmetic.OBSERVED_LANES
            ):
                assert bias is None
                return source_arithmetic.cached_projection(x, weight)
            decoder_projection = name.endswith(
                (
                    ".in_proj_qkv",
                    ".in_proj_a",
                    ".in_proj_b",
                    ".in_proj_z",
                    ".mlp.gate_proj",
                    ".mlp.up_proj",
                    ".mlp.down_proj",
                    ".linear_attn.out_proj",
                    ".self_attn.q_proj",
                    ".self_attn.k_proj",
                    ".self_attn.v_proj",
                    ".self_attn.o_proj",
                )
            )
            if name.startswith("model.vision3d.") or decoder_projection:
                slices = (
                    2
                    if rows >= 13824 and name.endswith((".in_proj_a", ".in_proj_b"))
                    else 3
                    if rows >= 13824 and name.endswith(".mlp.down_proj")
                    else 1
                )
                sliced = name.endswith(
                    (
                        ".mlp.down_proj",
                        ".linear_attn.out_proj",
                        ".self_attn.k_proj",
                        ".self_attn.v_proj",
                        ".self_attn.o_proj",
                    )
                )
                return source_arithmetic.projection(
                    x,
                    weight,
                    bias=bias,
                    slices=slices,
                    sliced=sliced,
                )
        if self.accumulate_float32:
            weight = weight.astype(mx.float32)
            if bias is not None:
                bias = bias.astype(mx.float32)
        return x @ weight.T if bias is None else mx.addmm(bias, x, weight.T)

    def norm(self, x: Array, name: str, eps: float = 1e-5) -> Array:
        """Apply the named LayerNorm in FP32 opmath, then cast to the working dtype.

        Returns FP32 for the profile that accumulates in FP32, otherwise the input
        dtype. Widths 864 and 2304 under ``source_bfloat16`` use the source-exact kernel.
        """
        if x.dtype == mx.bfloat16 and not self.accumulate_float32 and x.shape[-1] in (864, 2304):
            return source_arithmetic.layernorm(
                x,
                self.weights[name + ".weight"],
                self.weights[name + ".bias"],
                eps,
            )
        # CUDA LayerNorm keeps statistics and affine arithmetic in opmath FP32.
        # MLX's BF16 affine path rounds before the source's single output cast.
        dtype = mx.float32 if self.accumulate_float32 else x.dtype
        return mx.fast.layer_norm(
            x.astype(mx.float32),
            self.weights[name + ".weight"].astype(mx.float32),
            self.weights[name + ".bias"].astype(mx.float32),
            eps,
        ).astype(dtype)

    def rms(self, x: Array, name: str) -> Array:
        """Apply the named Qwen RMSNorm in FP32 with epsilon 1e-6.

        The learned weight is a zero-centered offset applied as ``1 + w``. Returns FP32
        for the profile that accumulates in FP32, otherwise the input dtype.
        """
        # Qwen's learned weight is a zero-centered offset; addition is FP32.
        fp = x.astype(mx.float32)
        if self.precision == "source_bfloat16" and x.shape[-1] in (128, 256, 2560):
            value = fp * source_arithmetic.rsqrt(source_arithmetic.mean_square(fp * fp) + 1e-6)
        else:
            value = fp * mx.rsqrt(mx.mean(fp * fp, axis=-1, keepdims=True) + 1e-6)
        dtype = mx.float32 if self.accumulate_float32 else x.dtype
        return (value * (1 + self.weights[name + ".weight"].astype(mx.float32))).astype(dtype)

    @staticmethod
    def attention(q: Array, k: Array, v: Array, *, causal: bool, offset: int = 0) -> Array:
        """Scaled dot-product attention on ``(B, heads, tokens, head_dim)`` arrays.

        ``k`` and ``v`` may have fewer heads than ``q`` (grouped-query attention).
        With ``causal``, query ``i`` attends to keys ``0..offset + i``, where
        ``offset`` is the number of cached keys before the first query.
        """
        if q.dtype == mx.bfloat16:
            return source_arithmetic.attention(q, k, v, causal=causal, offset=offset)
        # Bound the materialized score workspace even when a backend cannot fuse SDPA.
        outputs: list[Array] = []
        for start in range(0, q.shape[2], 256):
            end = min(start + 256, q.shape[2])
            mask = None
            if causal:
                mask = (
                    mx.arange(k.shape[2])[None, :]
                    <= mx.arange(offset + start, offset + end)[:, None]
                )
            result = mx.fast.scaled_dot_product_attention(
                q[:, :, start:end],
                k,
                v,
                scale=q.shape[-1] ** -0.5,
                mask=mask,
            )
            mx.eval(result)
            outputs.append(result)
        return mx.concatenate(outputs, axis=2)

    def vision(self, pixels: Array) -> tuple[Array, Array]:
        """Encode one CT crop into visual features and decoder-space embeddings.

        Args:
            pixels: ``(1, 1, 192, 192, 192)`` array, laid out ``(B, C, Z, Y, X)``.
                Values are expected to be HU clipped to [-1000, 1000] and divided
                by 1000; only the shape is checked.

        Returns:
            ``(features, embeddings)`` with shapes ``(1, 13824, 864)`` and
            ``(1, 13824, 2560)``. Tokens follow the 24 x 24 x 24 patch grid in
            ``(Z, Y, X)`` row-major order, one per 8 x 8 x 8 patch.

        Raises:
            InvalidInputError: ``pixels`` has any other shape.
        """
        if pixels.shape != (1, 1, 192, 192, 192):
            raise InvalidInputError("NV-Reason-CT vision expects a single 192-cubed CZYX CT")
        prefix = "model.vision3d.sub_vision"
        # Non-overlapping Conv3d is exactly a flattened patch projection.
        x = mx.reshape(pixels.astype(self.dtype), (1, 1, 24, 8, 24, 8, 24, 8))
        x = mx.reshape(mx.transpose(x, (0, 2, 4, 6, 1, 3, 5, 7)), (1, 13824, 512))
        weight = mx.reshape(self.weights[prefix + ".down_projection.proj.weight"], (864, 512))
        # CUDA Conv3d rounds its convolution before adding the channel bias.
        x = (
            source_arithmetic.convolution(x, weight)
            if self.precision == "source_bfloat16"
            else x @ weight.T
        ) + self.weights[prefix + ".down_projection.proj.bias"]
        if self.accumulate_float32:
            x = x.astype(mx.float32)
        for i in range(16):
            p = f"{prefix}.eva.blocks.{i}"
            h = self.norm(x, p + ".norm1")
            q, k, v = [
                mx.transpose(
                    mx.reshape(self.linear(h, p + f".attn.{s}_proj"), (1, -1, 12, 72)), (0, 2, 1, 3)
                )
                for s in ("q", "k", "v")
            ]

            def rotate(t: Array) -> Array:
                dtype = t.dtype
                t = t.astype(mx.float32)
                pairs = mx.reshape(t, (*t.shape[:-1], 36, 2))
                rotated = mx.reshape(mx.stack([-pairs[..., 1], pairs[..., 0]], axis=-1), t.shape)
                return (t * self.vision_cos + rotated * self.vision_sin).astype(dtype)

            q, k = rotate(q), rotate(k)
            if self.accumulate_float32:
                q, k, v = (t.astype(mx.float32) for t in (q, k, v))
            h = mx.reshape(
                mx.transpose(self.attention(q, k, v, causal=False), (0, 2, 1, 3)), (1, -1, 864)
            )
            h = self.linear(self.norm(h, p + ".attn.norm"), p + ".attn.proj")
            scale = self.weights[p + ".gamma_1"]
            if self.accumulate_float32:
                scale, h = scale.astype(mx.float32), h.astype(mx.float32)
            x = x + scale * h
            h = self.norm(x, p + ".norm2")
            # Feed-forward rows are evaluated in 512-row chunks to bound workspace memory.
            chunks: list[Array] = []
            for start in range(0, h.shape[1], 512):
                part = h[:, start : start + 512]
                part = self.silu(self.linear(part, p + ".mlp.fc1_g")) * self.linear(
                    part, p + ".mlp.fc1_x"
                )
                part = self.linear(self.norm(part, p + ".mlp.norm"), p + ".mlp.fc2")
                mx.eval(part)
                chunks.append(part)
            h = mx.concatenate(chunks, axis=1)
            scale = self.weights[p + ".gamma_2"]
            if self.accumulate_float32:
                scale, h = scale.astype(mx.float32), h.astype(mx.float32)
            x = x + scale * h
            mx.eval(x)
        features = self.norm(x, prefix + ".eva.norm")
        p = "model.vision3d.merger"
        embeddings = self.linear(
            self.gelu(self.linear(self.norm(features, p + ".norm", 1e-6), p + ".linear_fc1")),
            p + ".linear_fc2",
        )
        mx.eval(features, embeddings)
        return features, embeddings

    def rotary(self, x: Array, positions: Array) -> Array:
        """Apply multimodal rotary embedding to ``x`` of shape ``(B, heads, T, 256)``.

        ``positions`` is ``(3, B, T)``, one MRoPE index row per visual grid axis (see
        ``VolumePrompt.position_ids``). Only the first 64 channels of each head are
        rotated; the rest pass through.
        """
        phase = positions.astype(mx.float32)[..., None] * self.text_inv_freq
        # Interleaved temporal/height/width sections 11/11/10, exactly as Qwen3.5.
        channels = [0 if j % 3 == 0 or j >= 30 and j % 3 == 2 else j % 3 for j in range(32)]
        freqs = mx.stack([phase[axis, ..., j] for j, axis in enumerate(channels)], axis=-1)
        freqs = mx.concatenate([freqs, freqs], axis=-1)
        cos, sin = mx.cos(freqs).astype(x.dtype)[:, None], mx.sin(freqs).astype(x.dtype)[:, None]
        head, tail = x[..., :64], x[..., 64:]
        rotated = mx.concatenate([-head[..., 32:], head[..., :32]], axis=-1)
        return mx.concatenate([head * cos + rotated * sin, tail], axis=-1)

    def full_attention(
        self,
        x: Array,
        p: str,
        positions: Array,
        state: dict[str, Array],
        *,
        cache_offset: int | None = None,
        chunk_size: int = 0,
    ) -> Array:
        """Gated causal attention (16 query heads, 4 KV heads, head dim 256).

        ``x`` is ``(B, T, 2560)``; ``positions`` is ``(3, B, T)``. ``state`` holds
        the layer's ``k``/``v`` cache and is extended in place with this call's keys
        and values. The attention output is multiplied by the sigmoid of a per-channel
        gate produced alongside the queries.
        """
        batch, length, _ = x.shape
        q, gate = mx.split(
            mx.reshape(self.linear(x, p + ".q_proj"), (batch, length, 16, 512)), 2, axis=-1
        )
        q = mx.transpose(self.rms(q, p + ".q_norm"), (0, 2, 1, 3))
        k = mx.transpose(
            self.rms(
                mx.reshape(self.linear(x, p + ".k_proj"), (batch, length, 4, 256)), p + ".k_norm"
            ),
            (0, 2, 1, 3),
        )
        v = mx.transpose(
            mx.reshape(self.linear(x, p + ".v_proj"), (batch, length, 4, 256)), (0, 2, 1, 3)
        )
        q, k = self.rotary(q, positions), self.rotary(k, positions)
        offset = 0
        if chunk_size:
            offset = 0 if cache_offset is None else cache_offset
            required = offset + length
            capacity = cache_capacity(required, chunk_size)
            if "k" not in state:
                state["k"] = mx.zeros((batch, 4, capacity, 256), dtype=k.dtype)
                state["v"] = mx.zeros((batch, 4, capacity, 256), dtype=v.dtype)
            elif required > state["k"].shape[2]:
                extra = capacity - state["k"].shape[2]
                for name in ("k", "v"):
                    state[name] = mx.concatenate(
                        [state[name], mx.zeros((batch, 4, extra, 256), dtype=state[name].dtype)],
                        axis=2,
                    )
            state["k"][:, :, offset:required] = k
            state["v"][:, :, offset:required] = v
            k, v = state["k"][:, :, :required], state["v"][:, :, :required]
        elif "k" in state:
            offset = state["k"].shape[2]
            k, v = mx.concatenate([state["k"], k], axis=2), mx.concatenate([state["v"], v], axis=2)
        if not chunk_size:
            state.update(k=k, v=v)
        h = mx.reshape(
            mx.transpose(
                self.attention(q, k, v.astype(q.dtype), causal=True, offset=offset), (0, 2, 1, 3)
            ),
            (batch, length, -1),
        )
        return self.linear(h * self.sigmoid(mx.reshape(gate, (batch, length, -1))), p + ".o_proj")

    def linear_attention(self, x: Array, p: str, state: dict[str, Array]) -> Array:
        """Gated-delta linear attention (16 Q/K heads, 32 value heads, head dim 128).

        ``x`` is ``(B, T, 2560)``. ``state`` holds the layer's ``conv`` and
        ``recurrent`` entries (see ``DecoderCache``) and is updated in place. A
        sequence of more than one token uses the chunked prefill; a single token
        uses the one-step recurrence kernel.
        """
        batch, length, _ = x.shape
        source_bf16 = self.precision == "source_bfloat16"
        exponential = source_arithmetic.exp if source_bf16 else mx.exp
        qkv = self.linear(x, p + ".in_proj_qkv")
        previous = state.get("conv", mx.zeros((batch, 3, 8192), dtype=qkv.dtype))
        conv_input = mx.concatenate([previous, qkv], axis=1)
        state["conv"] = mx.contiguous(conv_input[:, -3:])
        qkv = self.silu(
            mx.conv1d(
                conv_input,
                mx.transpose(self.weights[p + ".conv1d.weight"].astype(qkv.dtype), (0, 2, 1)),
                groups=8192,
            )
        )
        q, k, v = mx.split(qkv, [2048, 4096], axis=-1)
        q, k = mx.reshape(q, (batch, length, 16, 128)), mx.reshape(k, (batch, length, 16, 128))
        v = mx.reshape(v, (batch, length, 32, 128))
        if self.accumulate_float32:
            q, k = q.astype(mx.float32), k.astype(mx.float32)
        normalized: list[Array] = []
        for value in (q, k):
            squared = value * value
            # ATen sums BF16 squares in FP32 opmath, then returns BF16. MLX's
            # BF16 reduction accumulates differently; preserve both cast points.
            if self.precision == "source_bfloat16":
                squared = squared.astype(mx.float32)
            norm = (
                source_arithmetic.sum_square(squared)
                if source_bf16
                else mx.sum(squared, axis=-1, keepdims=True)
            ).astype(value.dtype)
            normalized.append((value * mx.rsqrt(norm + 1e-6)).astype(mx.float32))
        q, k = normalized
        q = q * (128**-0.5)
        a = self.linear(x, p + ".in_proj_a").astype(mx.float32)
        beta = self.sigmoid(self.linear(x, p + ".in_proj_b")).astype(mx.float32)
        a = a + self.weights[p + ".dt_bias"].astype(mx.float32)
        # ATen Softplus uses log1p(exp(x)) below its explicit threshold of 20.
        logarithm = source_arithmetic.log1p if source_bf16 else mx.log1p
        softplus = mx.where(a > 20, a, logarithm(exponential(a)))
        log_decay = -exponential(self.weights[p + ".A_log"].astype(mx.float32)) * softplus
        recurrent = state.get("recurrent")
        if recurrent is None:
            recurrent = mx.zeros((batch, 32, 128, 128), dtype=mx.float32)
        recurrent = recurrent.astype(mx.float32)
        # Both prefill and single-token recurrence execute on Metal.
        if length > 1:
            h, recurrent = chunk_delta(
                q,
                k,
                v,
                log_decay,
                beta,
                recurrent,
                source_bfloat16=source_bf16,
            )
        else:
            recurrence = (
                source_arithmetic.gated_delta if source_bf16 else self.delta.gated_delta_kernel
            )
            h, recurrent = recurrence(
                q,
                k,
                v.astype(mx.float32),
                exponential(log_decay),
                beta,
                recurrent,
            )
        # The pinned cache inherits convolution dtype when storing FP32 recurrence.
        state["recurrent"] = (
            recurrent.astype(self.dtype) if self.precision == "source_bfloat16" else recurrent
        )
        h = h.astype(x.dtype)
        fp = h.astype(mx.float32)
        inverse = (
            source_arithmetic.rsqrt(source_arithmetic.mean_square(fp * fp) + 1e-6)
            if source_bf16
            else mx.rsqrt(mx.mean(fp * fp, axis=-1, keepdims=True) + 1e-6)
        )
        h = (fp * inverse).astype(x.dtype)
        h = h * self.weights[p + ".norm.weight"]
        z = mx.reshape(self.linear(x, p + ".in_proj_z"), (batch, length, 32, 128))
        activation = source_arithmetic.silu if source_bf16 else self.silu
        h = (h.astype(mx.float32) * activation(z.astype(mx.float32))).astype(x.dtype)
        return self.linear(mx.reshape(h, (batch, length, -1)), p + ".out_proj")

    def embed(self, ids: Array) -> Array:
        """Look up token embeddings for integer ``ids``."""
        value = self.weights["model.language_model.embed_tokens.weight"][ids]
        return value.astype(mx.float32) if self.accumulate_float32 else value

    def decode(
        self, embeddings: Array, positions: Array, cache: DecoderCache | None = None
    ) -> tuple[Array, DecoderCache]:
        """Return all final hidden states; callers project only consumed tokens.

        ``embeddings`` is ``(B, T, 2560)`` and ``positions`` the matching ``(3, B, T)``
        MRoPE indices. A new cache is created when none is given; otherwise the
        given cache is extended in place and returned with its ``offset`` advanced
        by ``T``.
        """
        if cache is None:
            cache = DecoderCache()
        x = embeddings.astype(mx.float32 if self.accumulate_float32 else self.dtype)
        for i in range(32):
            p = f"model.language_model.layers.{i}"
            h = self.rms(x, p + ".input_layernorm")
            # Every fourth layer uses full attention; the others use gated-delta
            # linear attention.
            if (i + 1) % 4:
                h = self.linear_attention(h, p + ".linear_attn", cache.layers[i])
            else:
                h = self.full_attention(
                    h,
                    p + ".self_attn",
                    positions,
                    cache.layers[i],
                    cache_offset=cache.offset,
                    chunk_size=cache.kv_chunk_size,
                )
            x = x + h
            h = self.rms(x, p + ".post_attention_layernorm")
            # Feed-forward rows are evaluated in chunks to bound peak workspace memory.
            chunks: list[Array] = []
            for start in range(0, h.shape[1], 256):
                part = h[:, start : start + 256]
                part = self.linear(
                    self.silu(
                        self.linear(
                            part,
                            p + ".mlp.gate_proj",
                            source_length=h.shape[1],
                        )
                    )
                    * self.linear(part, p + ".mlp.up_proj", source_length=h.shape[1]),
                    p + ".mlp.down_proj",
                    source_length=h.shape[1],
                )
                mx.eval(part)
                chunks.append(part)
            x = x + mx.concatenate(chunks, axis=1)
            mx.eval(x, *cache.layers[i].values())
        cache.offset += embeddings.shape[1]
        x = self.rms(x, "model.language_model.norm")
        mx.eval(x)
        return x, cache

    def logits(self, hidden: Array) -> Array:
        """Project hidden states to vocabulary logits with the tied embedding matrix."""
        weight = self.weights["model.language_model.embed_tokens.weight"]
        if self.accumulate_float32:
            weight = weight.astype(mx.float32)
        return hidden @ weight.T
