"""LLGuidance constraints that keep a response within explicit caller-provided choices."""

from __future__ import annotations

import json
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

from mlx_reason_ct.errors import InvalidInputError, MissingDependencyError, ModelExecutionError
from mlx_reason_ct.runtime import import_mlx

if TYPE_CHECKING:
    from llguidance import LLMatcher, LLTokenizer

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


@lru_cache(maxsize=1)
def _vocabulary(
    path: Path, modified_ns: int, n_vocab: int, eos_ids: tuple[int, ...]
) -> tuple[LLTokenizer, frozenset[int]]:
    """LLGuidance's tokenizer and the special IDs of one ``tokenizer.json``.

    Loading takes most of a second for the checkpoint's vocabulary, so the last
    one stays loaded for the next constraint; ``modified_ns`` retires it when the
    file is replaced.
    """
    from llguidance import LLTokenizer

    tokenizer = LLTokenizer(str(path), n_vocab=n_vocab, eos_token=list(eos_ids))
    return tokenizer, special_token_ids(path)


class ChoiceConstraint:
    """Keep greedy selection within a finite caller-provided response language.

    LLGuidance owns parsing and next-token masks. Each choice is a string literal,
    so the model may spell it with any tokens: a token that merges across a
    boundary, such as ``=l`` before ``iver`` or ``ung``, does not commit it to the
    one choice whose canonical tokenization starts that way. LLGuidance cannot
    spell its special tokens, which include Qwen's tool tags, so those stay token
    rules.

    The model decides only where the remaining choices diverge. :meth:`forced`
    returns the tokens they all share, which need no logits, and :meth:`sample`
    takes the greatest legal logit. Raw logits are never changed.
    """

    def __init__(self, tokenizer_path: Path, choices: Sequence[str]) -> None:
        """Validate 1-64 nonempty choices and defer the tokenizer/grammar to first use.

        Duplicate choices are dropped while the supplied order is kept. Rejects
        non-text, empty or over-4096-character choices, and requires the
        ``tools`` extra for LLGuidance.
        """
        self.choices = validate_response_choices(choices)
        try:
            from llguidance import LLMatcher
        except ImportError as error:
            raise MissingDependencyError(
                "Constrained decoding requires mlx-reason-ct[tools]"
            ) from error
        self._matcher_type = LLMatcher
        self._path = tokenizer_path
        self._matcher: LLMatcher | None = None
        self._tokenizer: LLTokenizer | None = None
        self._written = b""
        self._decided = False
        #: The model's probability of the selected token path among the permitted
        #: ones: the product, over the tokens it had to choose, of the softmax across
        #: the legal tokens.
        self.probability = 1.0
        #: The model's probability of each choice, set at the first decision when
        #: every legal token there continues exactly one choice; otherwise ``None``.
        self.probabilities: dict[str, float] | None = None

    def _ready(self, vocabulary: int, eos_ids: tuple[int, ...]) -> LLMatcher:
        """Tokenize the choices and build their grammar on first use."""
        if self._matcher is not None:
            return self._matcher
        tokenizer, special = _vocabulary(
            self._path, self._path.stat().st_mtime_ns, vocabulary, eos_ids
        )
        alternatives: list[str] = []
        for choice in self.choices:
            ids = tokenizer.tokenize_str(choice, parse_special=True)
            if any(token in eos_ids for token in ids):
                raise InvalidInputError("Response choices cannot contain an EOS token")
            # Tokenizer normalization or missing vocabulary entries must not
            # change the requested string. Match report decoding, which removes
            # the tokens tokenizer.json marks special. LLGuidance's own flag also
            # covers Qwen's tool tags, which decoding keeps, so it is not used here.
            visible = [token for token in ids if token not in special]
            if not ids or tokenizer.decode_str(visible) != choice:
                raise InvalidInputError("Response choices must round-trip through the tokenizer")
            parts: list[str] = []
            text: list[int] = []
            for token in [*ids, None]:
                if token is not None and not tokenizer.is_special_token(token):
                    text.append(token)
                    continue
                if text:
                    parts.append(json.dumps(tokenizer.decode_str(text), ensure_ascii=False))
                    text = []
                if token is not None:
                    parts.append(f"<[{token}]>")
            alternatives.append(" ".join(parts))
        grammar = self._matcher_type.grammar_from_lark("start: " + " | ".join(alternatives))
        self._tokenizer = tokenizer
        self._matcher = self._matcher_type(tokenizer, grammar)
        return self._matcher

    def _legal(self, matcher: LLMatcher, vocabulary: int) -> list[int]:
        """IDs the grammar permits next, ascending."""
        mask = np.frombuffer(matcher.compute_logit_bias(), dtype=np.uint8) == 200
        if matcher.is_error() or len(mask) != vocabulary or not np.any(mask):
            raise ModelExecutionError(f"Invalid constrained-token mask: {matcher.get_error()}")
        return cast(list[int], np.flatnonzero(mask).tolist())

    def _consume(self, matcher: LLMatcher, tokens: list[int]) -> None:
        # One at a time: only consume_token accepts the EOS that ends a choice.
        if not all(matcher.consume_token(token) for token in tokens):
            raise ModelExecutionError(
                f"Constraint rejected selected token: {matcher.get_error()}"
            )
        assert self._tokenizer is not None
        self._written += self._tokenizer.decode_bytes(tokens)

    def forced(self, vocabulary: int, eos_ids: tuple[int, ...]) -> list[int]:
        """Consume and return the tokens the grammar leaves no choice about.

        The run stops before the next token the model must choose. After a complete
        choice only the end of the turn can follow, so the run then ends with the
        first pinned EOS ID and nothing more needs decoding.

        Args:
            vocabulary: Size of the model's logit vector.
            eos_ids: The model's EOS IDs, turn end first.
        """
        matcher = self._ready(vocabulary, eos_ids)
        tokens: list[int] = []
        while True:
            # LLGuidance tokenizes text every remaining choice shares canonically.
            run = matcher.compute_ff_tokens()
            if not run:
                legal = self._legal(matcher, vocabulary)
                if all(token in eos_ids for token in legal):
                    return [*tokens, next(token for token in eos_ids if token in legal)]
                if len(legal) != 1:
                    return tokens
                run = legal
            self._consume(matcher, run)
            tokens += run

    def sample(self, logits: Array, eos_ids: tuple[int, ...]) -> int:
        """Select the greatest legal logit and consume that token.

        Updates ``probability``, and sets ``probabilities`` at the first position
        with more than one legal token when the choices allow it.
        """
        matcher = self._ready(logits.shape[-1], eos_ids)
        legal = self._legal(matcher, logits.shape[-1])
        mx = import_mlx()
        scores = np.asarray(logits[mx.array(legal)].astype(mx.float32), dtype=np.float64)
        weights = np.exp(scores - scores.max())
        weights /= weights.sum()
        # Ascending IDs keep argmax's first-maximum tie rule over the whole vocabulary.
        best = int(np.argmax(scores))
        if len(legal) > 1 and not self._decided:
            self._decided = True
            self.probabilities = self._choice_probabilities(
                legal, cast(list[float], weights.tolist()), eos_ids
            )
        self.probability *= float(weights[best])
        self._consume(matcher, [legal[best]])
        return legal[best]

    def _choice_probabilities(
        self, legal: list[int], weights: list[float], eos_ids: tuple[int, ...]
    ) -> dict[str, float] | None:
        """Probability of each choice when every legal token continues exactly one.

        Several tokens can start the same choice (``Yes``, ``Ye`` and ``Y``); their
        shares add up. A token that leaves two choices open, such as a shared first
        letter, would need the later decisions of every branch, so the result is
        then ``None``.
        """
        assert self._tokenizer is not None
        remaining = [choice.encode() for choice in self.choices]
        if any(not text.startswith(self._written) for text in remaining):
            return None
        remaining = [text[len(self._written) :] for text in remaining]
        totals = dict.fromkeys(self.choices, 0.0)
        for token, weight in zip(legal, weights, strict=True):
            spelled = self._tokenizer.decode_bytes([token])
            continued = [
                choice
                for choice, text in zip(self.choices, remaining, strict=True)
                if (not text if token in eos_ids else bool(spelled) and text.startswith(spelled))
            ]
            if len(continued) != 1:
                return None
            totals[continued[0]] += weight
        return totals
