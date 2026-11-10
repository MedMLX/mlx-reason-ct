"""Constrained decoding must preserve the caller's literal response choices."""

import json
from pathlib import Path

import pytest

from mlx_reason_ct.constrained import ChoiceConstraint
from mlx_reason_ct.errors import InvalidInputError
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
