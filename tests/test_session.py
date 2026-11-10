"""A fixed-CT session must not reuse embeddings after its source changes."""

import json
import os
from collections.abc import Sequence
from importlib import import_module
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from mlx_reason_ct import api
from mlx_reason_ct._host_types import Nibabel
from mlx_reason_ct._native_types import Array
from mlx_reason_ct.ct_crop import CTCrop
from mlx_reason_ct.errors import InvalidInputError, ModelExecutionError
from mlx_reason_ct.mlx_model import DecoderCache
from mlx_reason_ct.mlx_weights import SHARDS
from mlx_reason_ct.nifti import load_ct
from mlx_reason_ct.processor_mlx import VolumePrompt
from mlx_reason_ct.runtime import import_mlx

mx = import_mlx()
nib = cast(Nibabel, import_module("nibabel"))


class SmallProcessor:
    """Use real NIfTI input with a tiny deterministic tokenizer boundary."""

    def image(self, source: Path, region: str | None) -> tuple[np.ndarray, CTCrop]:
        values, affine = load_ct(source)
        crop = CTCrop(values, affine, affine, affine, values.shape, ((0, 0),) * 3, (0, 0, 0), None)
        return values[None, None], crop

    def prompt(
        self, question: str, enable_thinking: bool, history: Sequence[tuple[str, str]]
    ) -> VolumePrompt:
        return VolumePrompt(
            question,
            np.array([[0, 0]]),
            np.zeros((3, 1, 2), dtype=np.int64),
            np.ones((1, 2), dtype=np.int64),
            np.array([[1, 0]]),
            0,
            0,
        )

    def decode(self, tokens: list[int]) -> str:
        return "finding" if 0 in tokens else ""


class SmallModel:
    """Stand in for external weights while exercising the real generation lifecycle."""

    dtype = mx.float32
    eos_ids = (1,)

    def __init__(self) -> None:
        self._steps = 0

    def vision(self, pixels: Array) -> tuple[Array, Array]:
        embeddings = mx.zeros((1, 1, 1))
        return embeddings, embeddings

    def release_vision(self) -> None:
        """Implement the prepared model's release boundary; this fixture owns no visual weights."""

    def embed(self, ids: Array) -> Array:
        return mx.zeros((*ids.shape, 1))

    def decode(
        self, embeddings: Array, positions: Array, cache: DecoderCache | None = None
    ) -> tuple[Array, DecoderCache]:
        if cache is None:
            cache = DecoderCache()
        if cache.offset == 0:
            self._steps = 0
        cache.offset += embeddings.shape[1]
        return embeddings, cache

    def logits(self, hidden: Array) -> Array:
        self._steps += 1
        return mx.array([[[2.0, 0.0] if self._steps == 1 else [0.0, 2.0]]])


def test_session_rejects_source_rewrite_with_preserved_size_and_mtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "ct.nii"
    nib.save(nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.float32), np.eye(4)), source)
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    def load_assets(model_dir: Path) -> SmallProcessor:
        return SmallProcessor()

    def load_model(model_dir: Path, *, precision: str) -> SmallModel:
        return SmallModel()

    monkeypatch.setattr(api, "_load_assets", load_assets)
    monkeypatch.setattr(api, "_load_model", load_model)
    with api.CTSession(source, model_dir=model_dir, anatomy_region=None) as session:
        session.report(tmp_path / "first", max_new_tokens=2)
        session.report(tmp_path / "unchanged", max_new_tokens=2)
        original = source.stat()
        nib.save(nib.Nifti1Image(np.full((2, 2, 2), 4, dtype=np.float32), np.eye(4)), source)
        os.utime(source, ns=(original.st_atime_ns, original.st_mtime_ns))
        changed = source.stat()
        assert (changed.st_ino, changed.st_size, changed.st_mtime_ns) == (
            original.st_ino,
            original.st_size,
            original.st_mtime_ns,
        )
        with pytest.raises(InvalidInputError, match="CT file changed"):
            session.report(tmp_path / "changed", max_new_tokens=2)


@pytest.mark.parametrize("preexisting", [False, True])
def test_session_restores_report_set_when_run_record_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, preexisting: bool
) -> None:
    source = tmp_path / "ct.nii"
    nib.save(nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.float32), np.eye(4)), source)
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    def load_assets(model_dir: Path) -> SmallProcessor:
        return SmallProcessor()

    def load_model(model_dir: Path, *, precision: str) -> SmallModel:
        return SmallModel()

    monkeypatch.setattr(api, "_load_assets", load_assets)
    monkeypatch.setattr(api, "_load_model", load_model)
    output_dir = tmp_path / "report"
    paths = tuple(output_dir / name for name in ("report.txt", "model_response.json", "run.json"))
    if preexisting:
        output_dir.mkdir()
        for path in paths:
            path.write_bytes(b"old output")

    def fail_run(path: Path, content: str, *, overwrite: bool) -> None:
        if path.name == "run.json":
            raise OSError("injected run record failure")
        raise AssertionError(f"unexpected output write: {path}")

    monkeypatch.setattr(api, "write_private_text", fail_run)
    with api.CTSession(source, model_dir=model_dir, anatomy_region=None) as session:
        with pytest.raises(OSError, match="injected run record failure"):
            session.report(output_dir, max_new_tokens=2, overwrite=True)

    for path in paths:
        if preexisting:
            assert path.read_bytes() == b"old output"
        else:
            assert not path.exists()


def test_session_keeps_failed_run_records_for_incomplete_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "ct.nii"
    nib.save(nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.float32), np.eye(4)), source)
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    def load_assets(model_dir: Path) -> SmallProcessor:
        return SmallProcessor()

    def load_model(model_dir: Path, *, precision: str) -> SmallModel:
        return SmallModel()

    monkeypatch.setattr(api, "_load_assets", load_assets)
    monkeypatch.setattr(api, "_load_model", load_model)
    output_dir = tmp_path / "report"
    with api.CTSession(source, model_dir=model_dir, anatomy_region=None) as session:
        with pytest.raises(ModelExecutionError, match="incomplete response"):
            session.report(output_dir, max_new_tokens=1)

    assert (output_dir / "report.txt").read_text(encoding="utf-8") == "finding\n"
    assert (output_dir / "model_response.json").is_file()
    assert json.loads((output_dir / "run.json").read_text(encoding="utf-8"))["status"] == "failed"


def test_session_does_not_publish_a_report_if_source_changes_during_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "ct.nii"
    nib.save(nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.float32), np.eye(4)), source)
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    class MutatingModel(SmallModel):
        def logits(self, hidden: Array) -> Array:
            if self._steps == 0:
                nib.save(
                    nib.Nifti1Image(np.full((2, 2, 2), 4, dtype=np.float32), np.eye(4)), source
                )
            return super().logits(hidden)

    def load_assets(model_dir: Path) -> SmallProcessor:
        return SmallProcessor()

    def load_model(model_dir: Path, *, precision: str) -> MutatingModel:
        return MutatingModel()

    monkeypatch.setattr(api, "_load_assets", load_assets)
    monkeypatch.setattr(api, "_load_model", load_model)
    output_dir = tmp_path / "report"
    with api.CTSession(source, model_dir=model_dir, anatomy_region=None) as session:
        with pytest.raises(InvalidInputError, match="CT file changed"):
            session.report(output_dir, max_new_tokens=2)
        assert not session.history
    assert not (output_dir / "run.json").exists()


@pytest.mark.parametrize("choice", ["a", "before</think>after", "a<answer></answer>"])
def test_constrained_report_preserves_the_literal_selected_choice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, choice: str
) -> None:
    """Exercise the actual tokenizer, grammar, preprocessing and report writer."""
    pytest.importorskip("llguidance")
    source = tmp_path / "ct.nii"
    nib.save(nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.float32), np.eye(4)), source)
    model_dir = tmp_path / "model"
    (model_dir / "mlx").mkdir(parents=True)
    for name in SHARDS:
        (model_dir / "mlx" / name).touch()
    for name in ("config.json", "generation_config.json"):
        (model_dir / name).write_text("{}")
    (model_dir / "image_processor_3d").mkdir()
    (model_dir / "image_processor_3d/preprocessor_config.json").write_text(
        json.dumps(
            {"final_grid_size": [24, 24, 24], "spatial_size": [192, 192, 192], "pixdim": [2, 2, 2]}
        )
    )
    (model_dir / "chat_template.jinja").write_text("<|image_pad|>")
    vocab = {character: index for index, character in enumerate(sorted(set(choice)))}
    vocab["[EOS]"] = len(vocab)
    vocab["<|image_pad|>"] = len(vocab)
    (model_dir / "tokenizer.json").write_text(
        json.dumps(
            {
                "version": "1.0",
                "added_tokens": [
                    {
                        "id": index,
                        "content": content,
                        "single_word": False,
                        "lstrip": False,
                        "rstrip": False,
                        "normalized": False,
                        "special": True,
                    }
                    for content in ("[EOS]", "<|image_pad|>")
                    for index in (vocab[content],)
                ],
                "decoder": {
                    "type": "ByteLevel",
                    "add_prefix_space": True,
                    "trim_offsets": True,
                    "use_regex": True,
                },
                "model": {"type": "BPE", "vocab": vocab, "merges": []},
            }
        )
    )

    class ChoiceModel(SmallModel):
        eos_ids = (vocab["[EOS]"],)

        def vision(self, pixels: Array) -> tuple[Array, Array]:
            embeddings = mx.zeros((1, 13824, 1))
            return embeddings, embeddings

        def logits(self, hidden: Array) -> Array:
            ids = [vocab[character] for character in choice] + list(self.eos_ids)
            target = ids[self._steps]
            self._steps += 1
            result = mx.full((1, 1, len(vocab)), -100.0)
            result[..., target] = 100.0
            return result

    def load_model(model_dir: Path, *, precision: str) -> ChoiceModel:
        return ChoiceModel()

    monkeypatch.setattr(api, "_load_model", load_model)
    output_dir = tmp_path / "report"
    result = api.generate_report(
        source,
        output_dir,
        model_dir=model_dir,
        anatomy_region=None,
        response_choices=[choice],
        max_new_tokens=len(choice) + 1,
    )
    assert result["status"] == "succeeded"
    assert (output_dir / "report.txt").read_text() == choice + "\n"
    response = json.loads((output_dir / "model_response.json").read_text())
    assert response["report"] == choice
