"""Model bundle admission for the published native safetensors shards."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from mlx_reason_ct._payloads import BundleManifest
from mlx_reason_ct.errors import InvalidInputError

REVISION = "386b93e034983f6c1fc841a43833a1b6a0cd9c13"
ENGINE = "nv_reason_ct_mlx.fp32.v1"
MANIFEST = "mlx/manifest.json"
SHARDS = (
    "vision.safetensors",
    "embedding.safetensors",
    "norm.safetensors",
    *(f"layer_{i:02d}.safetensors" for i in range(32)),
)
MODEL_FILES = (
    "config.json",
    "generation_config.json",
    "chat_template.jinja",
    "tokenizer.json",
    "image_processor_3d/preprocessor_config.json",
)
RUNTIME_FILES = ("LICENSE", *MODEL_FILES)


def require_runtime_assets(model_dir: Path) -> None:
    """Require actual processor files and native shards, without a conversion receipt.

    This checks file availability only. The processor parses its real configuration
    and the native loader retains its FP32 dtype checks. Source identity is not
    inferred from a sidecar, directory name, or conversion metadata.
    """
    for name in (*MODEL_FILES, *(f"mlx/{shard}" for shard in SHARDS)):
        if not (model_dir / name).is_file():
            raise InvalidInputError(f"NV-Reason-CT model asset missing: {name}")


def read_manifest(model_dir: Path) -> BundleManifest:
    """Read and check a converted bundle's manifest against the pinned conversion.

    Checks that the runtime files and every listed shard exist, and that engine,
    revision and dtype match the pinned values. Shard contents are not hashed or
    read.

    Raises:
        InvalidInputError: a runtime file or shard is missing, or the manifest
            provenance or shard inventory differs from the pinned conversion.
    """
    for name in RUNTIME_FILES:
        if not (model_dir / name).is_file():
            raise InvalidInputError(f"NV-Reason-CT runtime asset missing: {name}")
    payload: object = json.loads((model_dir / MANIFEST).read_text())
    if not isinstance(payload, dict):
        raise InvalidInputError("MLX-Reason-CT cache manifest must be an object")
    manifest = cast(BundleManifest, payload)
    if (
        manifest.get("engine") != ENGINE
        or manifest.get("revision") != REVISION
        or manifest.get("dtype") != "float32"
    ):
        raise InvalidInputError("MLX-Reason-CT cache provenance differs from the pinned conversion")
    inventory = cast(object, manifest.get("shards", []))
    if not isinstance(inventory, list | dict):
        raise InvalidInputError("MLX-Reason-CT cache shard inventory must be a list")
    # Bundles converted before shard hashes were dropped list them as mapping keys.
    shards = list(cast(list[str] | dict[str, str], inventory))
    if set(shards) != set(SHARDS):
        raise InvalidInputError("MLX-Reason-CT cache has an invalid shard inventory")
    for name in shards:
        if not (model_dir / "mlx" / name).is_file():
            raise InvalidInputError(f"MLX-Reason-CT cache shard missing: {name}")
    return manifest
