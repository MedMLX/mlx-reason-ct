"""Published bundle admission checks."""

import json
import sys
from importlib import import_module
from pathlib import Path

import pytest
from huggingface_hub import DryRunFileInfo
from medmlx_core.weights import weights_revision

from mlx_reason_ct import cli
from mlx_reason_ct import mlx_weights as weights
from mlx_reason_ct.errors import InvalidInputError


def download_inventory(revision: str) -> list[DryRunFileInfo]:
    """Return a complete Hub dry-run result for a cached synthetic bundle."""
    return [
        DryRunFileInfo(
            commit_hash=revision,
            file_size=0,
            filename=name,
            is_cached=True,
            local_path=name,
            will_download=False,
        )
        for name in (
            *weights.RUNTIME_FILES,
            weights.MANIFEST,
            *(f"mlx/{s}" for s in weights.SHARDS),
        )
    ]


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


@pytest.mark.parametrize(
    "corruption", ["shard", "inventory", "revision", "missing-manifest", "manifest-json"]
)
def test_bundle_rejects_corruption(tmp_path: Path, corruption: str) -> None:
    bundle = make_bundle(tmp_path / "bundle")
    manifest_path = bundle / "mlx/manifest.json"
    weights.read_manifest(bundle)  # The positive control reaches the same admission owner.
    if corruption == "shard":
        (bundle / "mlx/vision.safetensors").unlink()
    elif corruption == "missing-manifest":
        manifest_path.unlink()
    elif corruption == "manifest-json":
        manifest_path.write_text("{")
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

    def snapshot_download(
        repo_id: str, *, revision: str, local_dir: Path, dry_run: bool = False
    ) -> str | list[DryRunFileInfo]:
        """Stand in for the Hub with a complete bundle."""
        if dry_run:
            return download_inventory(revision)
        return str(make_bundle(Path(local_dir)))

    monkeypatch.setattr(import_module("huggingface_hub"), "snapshot_download", snapshot_download)
    monkeypatch.setattr(
        sys,
        "argv",
        ["mlx-reason-ct", "download", "--revision", revision, "--model-dir", str(model_dir)],
    )
    assert cli.main() == 0
    assert weights_revision(model_dir) == revision


@pytest.mark.parametrize("fault", ["missing-remote-shard", "stale-file", "pending-download"])
def test_download_cannot_label_a_mixed_bundle_as_the_requested_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    revision = "c" * 40
    bundle = make_bundle(tmp_path / "bundle")
    (bundle / ".medmlx-revision").write_text("b" * 40 + "\n")
    inventory = download_inventory(revision)
    if fault == "missing-remote-shard":
        inventory.pop()
    elif fault == "stale-file":
        inventory[-1].commit_hash = "b" * 40
    else:
        inventory[-1].will_download = True

    def snapshot_download(
        repo_id: str, *, revision: str, local_dir: Path, dry_run: bool = False
    ) -> str | list[DryRunFileInfo]:
        # A local_dir download can leave old files in place; only Hub metadata
        # distinguishes them from a complete download of the requested commit.
        return inventory if dry_run else str(local_dir)

    monkeypatch.setattr(import_module("huggingface_hub"), "snapshot_download", snapshot_download)
    monkeypatch.setattr(
        sys,
        "argv",
        ["mlx-reason-ct", "download", "--revision", revision, "--model-dir", str(bundle)],
    )
    with pytest.raises(InvalidInputError):
        cli.main()
    assert not (bundle / ".medmlx-revision").exists()
