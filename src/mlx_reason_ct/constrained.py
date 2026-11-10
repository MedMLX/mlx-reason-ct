"""LLGuidance constraints over canonical tokenizations of explicit response choices."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

from mlx_reason_ct.errors import InvalidInputError, MissingDependencyError, ModelExecutionError
from mlx_reason_ct.runtime import import_mlx

if TYPE_CHECKING:
    from llguidance import LLMatcher

    from mlx_reason_ct._native_types import Array


def validate_response_choices(choices: Sequence[str]) -> tuple[str, ...]:
    """Validate and deduplicate literal choices before downloading or loading weights."""
    raw = cast(object, choices)
    if isinstance(raw, str | bytes) or not isinstance(raw, Sequence):
        raise InvalidInputError("response_choices must contain 1 to 64 nonempty strings")
    values = cast(Sequence[object], raw)
    if not 1 <= len(values) <= 64:
        raise InvalidInputError("response_choices must contain 1 to 64 nonempty strings")
    if any(not isinstance(c, str) or not c or len(c) > 4096 for c in values):
        raise InvalidInputError("Each response choice must be text of length 1 to 4096")
    return tuple(dict.fromkeys(cast(Sequence[str], values)))


def special_token_ids(tokenizer_path: Path) -> frozenset[int]:
    """IDs that ``tokenizer.json`` marks special, which decoding with skip_special_tokens drops."""
    document = cast(dict[str, object], json.loads(tokenizer_path.read_text()))
    added = cast(list[dict[str, object]], document.get("added_tokens", []))
    return frozenset(cast(int, token["id"]) for token in added if token.get("special") is True)


class ChoiceConstraint:
    """Keep greedy MLX selection within a finite caller-provided response language.

    LLGuidance owns parsing and next-token masks. Numeric-token Lark rules
    preserve the checkpoint tokenizer's canonical encoding, including Qwen's
    added tool tokens; literal-byte grammars would reject those special tokens.
    Logits and argmax stay on Metal. Host work only constructs the allowed mask.
    """

    def __init__(self, tokenizer_path: Path, choices: Sequence[str]) -> None:
        """Validate 1-64 nonempty choices and defer the tokenizer/grammar to first sampling.

        Duplicate choices are dropped while the supplied order is kept. Rejects
        non-text, empty or over-4096-character choices, and requires the
        ``tools`` extra for LLGuidance.
        """
        self.choices = validate_response_choices(choices)
        try:
            from llguidance import LLMatcher, LLTokenizer
        except ImportError as error:
            raise MissingDependencyError(
                "Constrained decoding requires mlx-reason-ct[tools]"
            ) from error
        self._tokenizer_type = LLTokenizer
        self._matcher_type = LLMatcher
        self._path = tokenizer_path
        self._matcher: LLMatcher | None = None

    def sample(self, logits: Array, eos_ids: tuple[int, ...]) -> int:
        """Mask disallowed IDs, select the greatest legal logit, and consume that token."""
        if self._matcher is None:
            tokenizer = self._tokenizer_type(
                str(self._path),
                n_vocab=logits.shape[-1],
                eos_token=list(eos_ids),
            )
            special = special_token_ids(self._path)
            alternatives: list[str] = []
            for choice in self.choices:
                ids = tokenizer.tokenize_str(choice, parse_special=True)
                if any(token in eos_ids for token in ids):
                    raise InvalidInputError("Response choices cannot contain an EOS token")
                # Tokenizer normalization or missing vocabulary entries must not
                # change the requested string. Match report decoding, which removes
                # the tokens tokenizer.json marks special. LLGuidance's own flag also
                # covers Qwen's tool tags, which decoding keeps, so it is not used.
                visible = [token for token in ids if token not in special]
                if not ids or tokenizer.decode_str(visible) != choice:
                    raise InvalidInputError(
                        "Response choices must round-trip through the tokenizer"
                    )
                alternatives.append(" ".join(f"<[{token}]>" for token in ids))
            grammar = self._matcher_type.grammar_from_lark("start: " + " | ".join(alternatives))
            self._matcher = self._matcher_type(tokenizer, grammar)
        mask = np.frombuffer(self._matcher.compute_logit_bias(), dtype=np.uint8) == 200
        if self._matcher.is_error() or len(mask) != logits.shape[-1] or not np.any(mask):
            raise ModelExecutionError(
                f"Invalid constrained-token mask: {self._matcher.get_error()}"
            )
        mx = import_mlx()
        allowed = mx.where(mx.array(mask), logits, -float("inf"))
        token = int(cast(int, mx.argmax(allowed).item()))
        if not self._matcher.consume_token(token):
            raise ModelExecutionError(
                f"Constraint rejected selected token: {self._matcher.get_error()}"
            )
        return token
