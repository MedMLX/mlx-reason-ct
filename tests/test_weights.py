"""Published bundle admission checks."""

import json
import sys
from importlib import import_module
from pathlib import Path

import pytest
from medmlx_core.weights import weights_revision

from mlx_reason_ct import cli
from mlx_reason_ct import mlx_weights as weights
from mlx_reason_ct.errors import InvalidInputError


def make_bundle(directory: Path) -> Path:
    """Create a minimal bundle for manifest and asset admission tests."""
    directory.mkdir()
    for name in weights.RUNTIME_FILES:
        target = directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("synthetic " + name)
    shard_dir = directory / "mlx"
    shard_dir.mkdir()
    for name in weights.SHARDS:
        (shard_dir / name).write_bytes(b"placeholder")
    (shard_dir / "manifest.json").write_text(
        json.dumps(
            {
                "engine": weights.ENGINE,
                "revision": weights.REVISION,
                "dtype": "float32",
                "shards": list(weights.SHARDS),
            }
        )
    )
    return directory


@pytest.mark.parametrize("corruption", ["shard", "inventory", "revision"])
def test_bundle_rejects_corruption(tmp_path: Path, corruption: str) -> None:
    bundle = make_bundle(tmp_path / "bundle")
    manifest_path = bundle / "mlx/manifest.json"
    weights.read_manifest(bundle)  # The positive control reaches the same admission owner.
    if corruption == "shard":
        (bundle / "mlx/vision.safetensors").unlink()
    else:
        manifest = json.loads(manifest_path.read_text())
        if corruption == "inventory":
            manifest["shards"].remove("vision.safetensors")
        else:
            manifest["revision"] = "incorrect"
        manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(InvalidInputError):
        weights.read_manifest(bundle)


def test_download_records_the_fetched_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Record the downloaded commit so runs on the folder report it as weights_revision."""
    revision = "b" * 40
    model_dir = tmp_path / "models"

    def snapshot_download(repo_id: str, *, revision: str, local_dir: Path) -> str:
        """Stand in for the Hub with a complete bundle."""
        return str(make_bundle(Path(local_dir)))

    monkeypatch.setattr(import_module("huggingface_hub"), "snapshot_download", snapshot_download)
    monkeypatch.setattr(
        sys,
        "argv",
        ["mlx-reason-ct", "download", "--revision", revision, "--model-dir", str(model_dir)],
    )
    assert cli.main() == 0
    assert weights_revision(model_dir) == revision
