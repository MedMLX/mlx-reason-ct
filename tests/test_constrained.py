"""Constrained decoding must preserve the caller's literal response choices."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from mlx_reason_ct._native_types import Array
from mlx_reason_ct.api import generate
from mlx_reason_ct.constrained import ChoiceConstraint
from mlx_reason_ct.errors import InvalidInputError
from mlx_reason_ct.mlx_model import DecoderCache
from mlx_reason_ct.processor_mlx import VolumePrompt
from mlx_reason_ct.runtime import import_mlx

pytest.importorskip("llguidance")
mx = import_mlx()


@pytest.fixture
def tokenizer_path(tmp_path: Path) -> Path:
    """A real byte-level BPE tokenizer with a deliberately incomplete vocabulary."""
    path = tmp_path / "tokenizer.json"
    path.write_text(
        json.dumps(
            {
                "version": "1.0",
                "added_tokens": [
                    {
                        "id": 2,
                        "content": "x",
                        "single_word": False,
                        "lstrip": False,
                        "rstrip": False,
                        "normalized": False,
                        "special": True,
                    },
                    {
                        "id": 3,
                        "content": "[EOS]",
                        "single_word": False,
                        "lstrip": False,
                        "rstrip": False,
                        "normalized": False,
                        "special": True,
                    }
                ],
                "decoder": {
                    "type": "ByteLevel",
                    "add_prefix_space": True,
                    "trim_offsets": True,
                    "use_regex": True,
                },
                "model": {
                    "type": "BPE",
                    "vocab": {"a": 0, "b": 1, "x": 2, "[EOS]": 3},
                    "merges": [],
                },
            }
        )
    )
    return path


def test_choice_prefixes_and_eos_exclude_higher_scoring_forbidden_tokens(
    tokenizer_path: Path,
) -> None:
    constraint = ChoiceConstraint(tokenizer_path, ["a", "ab"])
    selected = [
        constraint.sample(mx.array(values, dtype=mx.float32), (3,))
        for values in ([1, 2, 9, 0], [0, 2, 9, 1], [0, 2, 9, 1])
    ]
    assert selected == [0, 1, 3]


def test_choices_decided_by_their_first_token_report_the_model_probability_of_each(
    tokenizer_path: Path,
) -> None:
    logits = mx.array([1, 2, 9, 0], dtype=mx.float32)
    constraint = ChoiceConstraint(tokenizer_path, ["b", "a"])
    constraint.sample(logits, (3,))
    # Softmax over the two permitted first tokens only: logit 2 for "b", 1 for "a".
    assert constraint.probabilities == pytest.approx(
        {"b": math.e / (1 + math.e), "a": 1 / (1 + math.e)}
    )
    shared_start = ChoiceConstraint(tokenizer_path, ["a", "ab"])
    shared_start.sample(logits, (3,))
    assert shared_start.probabilities is None


def test_token_merged_across_a_choice_boundary_keeps_both_choices_open(tmp_path: Path) -> None:
    """ "=lu" is spelled "=", "lu" and "=li" is spelled "=l", "i" by this tokenizer.

    A model that starts with "=l" has not chosen "=li": "u" may still follow.
    """
    vocab = {"=": 0, "l": 1, "u": 2, "i": 3, "lu": 4, "=l": 5, "[EOS]": 6}
    path = tmp_path / "tokenizer.json"
    document: dict[str, object] = {
        "version": "1.0",
        "added_tokens": [
            {
                "id": 6,
                "content": "[EOS]",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
        ],
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": True,
            "trim_offsets": True,
            "use_regex": True,
        },
        "model": {"type": "BPE", "vocab": vocab, "merges": ["l u", "= l"]},
    }
    path.write_text(json.dumps(document))
    constraint = ChoiceConstraint(path, ["=lu", "=li"])
    prefers_merged_token = mx.array([0, 0, 0, 0, 0, 9, 0], dtype=mx.float32)
    prefers_u = mx.array([0, 0, 9, 1, 0, 0, 0], dtype=mx.float32)
    assert constraint.sample(prefers_merged_token, (6,)) == vocab["=l"]
    assert constraint.sample(prefers_u, (6,)) == vocab["u"]
    assert constraint.forced(7, (6,)) == [vocab["[EOS]"]]


class CountingModel:
    """Prefers "a" until one generated token has been decoded, then "b"."""

    eos_ids = (3,)

    def __init__(self) -> None:
        self.decoded: list[int] = []

    def embed(self, ids: Array) -> Array:
        return mx.array(np.asarray(ids, dtype=np.float32)[..., None])

    def decode(
        self, embeddings: Array, positions: Array, cache: DecoderCache | None = None
    ) -> tuple[Array, DecoderCache]:
        if cache is None:
            return embeddings, DecoderCache(offset=embeddings.shape[1])
        self.decoded.append(int(np.asarray(embeddings, dtype=np.float32)[0, 0, 0]))
        cache.offset += 1
        return embeddings, cache

    def logits(self, hidden: Array) -> Array:
        return mx.array([[[9.0, 1.0, 0.0, 0.0] if not self.decoded else [1.0, 9.0, 0.0, 0.0]]])


def test_shared_tokens_are_emitted_without_logits_and_decoded_only_for_a_later_decision(
    tokenizer_path: Path,
) -> None:
    """Both choices start with "a"; the model then decides between "a" and "b"."""
    prompt = VolumePrompt(
        "",
        np.array([[0, 0]]),
        np.zeros((3, 1, 2), dtype=np.int64),
        np.ones((1, 2), dtype=np.int64),
        np.array([[1, 0]]),
        0,
        0,
    )
    model = CountingModel()
    tokens, metrics = generate(
        model,
        prompt,
        mx.zeros((1, 1, 1)),
        max_new_tokens=8,
        constraint=ChoiceConstraint(tokenizer_path, ["aab", "aba"]),
    )
    # The decision follows the shared "a", so it must use the logits after that
    # token (which prefer "b"), not the prompt's (which prefer "a").
    assert tokens == [0, 1, 0, 3]
    # Nothing after the decision is open, so only the shared token was decoded.
    assert model.decoded == [0]
    assert metrics["choice_probability"] == pytest.approx(math.exp(9) / (math.exp(9) + math.e))


def test_unrepresentable_choice_is_not_silently_shortened(tokenizer_path: Path) -> None:
    constraint = ChoiceConstraint(tokenizer_path, ["xZ"])
    with pytest.raises(InvalidInputError):
        constraint.sample(mx.array([0, 0, 5, 0], dtype=mx.float32), (3,))


def test_choice_cannot_disappear_when_report_decoding_skips_special_tokens(
    tokenizer_path: Path,
) -> None:
    constraint = ChoiceConstraint(tokenizer_path, ["x"])
    with pytest.raises(InvalidInputError):
        constraint.sample(mx.array([0, 0, 5, 0], dtype=mx.float32), (3,))


def test_added_token_that_decoding_keeps_is_a_valid_choice(tmp_path: Path) -> None:
    """Qwen's tool tags are added but not special tokens, so report decoding keeps them."""
    path = tmp_path / "tokenizer.json"
    document: dict[str, object] = {
        "version": "1.0",
        "added_tokens": [
            {
                "id": 2,
                "content": "<t>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": False,
            },
            {
                "id": 3,
                "content": "[EOS]",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            },
        ],
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": True,
            "trim_offsets": True,
            "use_regex": True,
        },
        "model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "<t>": 2, "[EOS]": 3}, "merges": []},
    }
    path.write_text(json.dumps(document))
    constraint = ChoiceConstraint(path, ["<t>a"])
    selected = [
        constraint.sample(mx.array(values, dtype=mx.float32), (3,))
        for values in ([9, 0, 1, 0], [0, 9, 1, 0], [0, 0, 1, 9])
    ]
    assert selected == [2, 0, 3]
