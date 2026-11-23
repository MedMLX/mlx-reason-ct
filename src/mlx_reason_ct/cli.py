"""Standalone bundle download, verification and offline inference."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping, Sequence
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, cast

from mlx_reason_ct._payloads import JsonValue
from mlx_reason_ct.api import PRECISION_PROFILES, CTSession, write_report
from mlx_reason_ct.errors import InvalidInputError
from mlx_reason_ct.mlx_weights import MANIFEST, RUNTIME_FILES, SHARDS, read_manifest
from mlx_reason_ct.response import THINK_END, write_private_text

if TYPE_CHECKING:
    from huggingface_hub import DryRunFileInfo

_COMMIT = re.compile(r"[0-9a-f]{40}")
# The download record medmlx_core.weights.weights_revision() reads.
_REVISION_FILE = ".medmlx-revision"


def main() -> int:
    """Run the selected download, verify, report or chat subcommand; return the exit status."""
    parser = argparse.ArgumentParser(prog="mlx-reason-ct", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="Download the FP32 bundle")
    download.add_argument(
        "--repo", default="MedMLX/MLX-Reason-CT", help="Hub repository (default: %(default)s)"
    )
    download.add_argument(
        "--revision", required=True, type=_commit, help="Pinned 40-character Hub commit"
    )
    download.add_argument("--model-dir", type=Path, required=True)
    verify = commands.add_parser(
        "verify", help="Check manifest metadata and required file presence"
    )
    verify.add_argument("--model-dir", type=Path, required=True)
    report = commands.add_parser("report", help="Generate a report from one local HU NIfTI CT")
    chat = commands.add_parser(
        "chat", help="Ask several questions about one CT; the model stays loaded"
    )
    for command in (report, chat):
        command.add_argument("--input", type=Path, required=True)
        command.add_argument("--model-dir", type=Path, required=True)
        command.add_argument("--output-dir", type=Path, required=True)
        command.add_argument(
            "--anatomy-region",
            choices=["chest", "abdomen", "none"],
            default="chest",
            help="'none' uses the pinned processor's own centered crop, with no "
            "chest/abdomen lung detection",
        )
        command.add_argument("--enable-thinking", action="store_true")
        command.add_argument("--max-new-tokens", type=int, default=1024)
        command.add_argument(
            "--precision", choices=tuple(PRECISION_PROFILES), default="fast-bf16"
        )
        command.add_argument(
            "--overwrite", action="store_true", help="Replace existing report files"
        )
    report.add_argument(
        "--prompt", help="Question or instruction (default: structured report for the region)"
    )
    report.add_argument(
        "--history",
        type=Path,
        help="JSON file holding a list of already-answered [question, answer] "
        "turns, oldest first, to continue a conversation about the same CT",
    )
    report.add_argument("--tools", type=Path, help="JSON file holding a list of function schemas")
    report.add_argument(
        "--tool-history",
        type=Path,
        help="JSON file holding alternating assistant/tool turns as {role, content} "
        "objects, ending with a tool result",
    )
    report.add_argument(
        "--tool-choice",
        choices=["required"],
        help="Constrain the answer to one call of a tool from --tools",
    )
    report.add_argument(
        "--response-choice",
        action="append",
        help="A complete permitted answer; repeat to let the model choose among several",
    )
    chat.add_argument(
        "--prefill-reuse",
        choices=["exact", "fast"],
        default="exact",
        help="'fast' answers follow-up questions sooner; its logits differ at rounding level",
    )
    args = parser.parse_args()
    if args.command == "download":
        _download(args.repo, args.revision, args.model_dir)
        print(args.model_dir)
    elif args.command == "verify":
        manifest = read_manifest(args.model_dir)
        print(
            json.dumps(
                {
                    "verified": True,
                    "revision": manifest["revision"],
                    "dtype": manifest["dtype"],
                    "shards": len(manifest["shards"]),
                }
            )
        )
    elif args.command == "chat":
        _chat(args)
    else:
        # Preserve untrusted JSON for the API's validators. Stringifying entries
        # would turn malformed values into plausible conversation text.
        history = cast("Sequence[tuple[str, str]] | None", _json_file(args.history))
        result = write_report(
            args.input,
            args.output_dir,
            model_dir=args.model_dir,
            prompt=args.prompt,
            anatomy_region=None if args.anatomy_region == "none" else args.anatomy_region,
            enable_thinking=args.enable_thinking,
            max_new_tokens=args.max_new_tokens,
            precision=args.precision,
            history=history,
            tools=cast("Sequence[Mapping[str, JsonValue]] | None", _json_file(args.tools)),
            tool_history=cast(
                "Sequence[Mapping[str, str]] | None", _json_file(args.tool_history)
            ),
            response_choices=args.response_choice,
            tool_choice=args.tool_choice,
            overwrite=args.overwrite,
        )
        print(json.dumps(result, indent=2))
    return 0


def _json_file(path: Path | None) -> object:
    return None if path is None else json.loads(path.read_text(encoding="utf-8"))


_EMPTY_ANSWER = "<answer></answer>"


def _shown(response: str, *, thinking: bool, complete: bool) -> str:
    """Return what the chat prints for the raw ``response`` generated so far.

    Reasoning is labelled and separated from the answer at ``</think>``. Trailing
    whitespace, and a tail that can still become the empty ``<answer></answer>``
    the saved report drops, are withheld until ``complete``, so the text returned
    for a longer response always starts with the text returned for a shorter one.
    """
    reasoning, closed, answer = response.partition(THINK_END)
    if not closed and not thinking:
        reasoning, answer = "", response
    elif not closed and not complete:
        # The closing tag may still be arriving.
        for length in range(len(THINK_END) - 1, 0, -1):
            if reasoning.endswith(THINK_END[:length]):
                reasoning = reasoning[:-length]
                break
    if complete:
        answer = re.sub(r"\s*<answer>\s*</answer>\s*$", "", answer)
    elif closed or not thinking:
        # Withhold the longest tail that, ignoring whitespace, starts the empty element.
        keep = tail = len(answer.rstrip())
        compact = ""
        while tail and len(compact) < len(_EMPTY_ANSWER):
            tail -= 1
            if not answer[tail].isspace():
                compact = answer[tail] + compact
                if _EMPTY_ANSWER.startswith(compact):
                    keep = tail
        answer = answer[:keep]
    text = f"[thinking] {reasoning.strip()}" if reasoning.strip() else ""
    if closed or not thinking:
        text += ("\n\n" if text else "") + answer.strip()
    return text


def _chat(args: argparse.Namespace) -> None:
    """Answer each line of standard input about one CT until an empty line or end of input.

    Earlier questions and answers are replayed with each new one, so follow-ups can
    refer back. Turn ``N`` writes its files to ``turn-NNN`` under the output folder.
    Each answer is printed as it is generated.
    """
    interactive = sys.stdin.isatty()
    if interactive:
        print("Ask about the CT. An empty line or Ctrl-D ends the conversation.")
    history: list[tuple[str, str]] = []
    with CTSession(
        image_path=args.input,
        weights_path=args.model_dir,
        anatomy_region=None if args.anatomy_region == "none" else args.anatomy_region,
        precision=args.precision,
        prefill_reuse=args.prefill_reuse,
    ) as session:
        while True:
            if interactive:
                print("> ", end="", flush=True)
            question = sys.stdin.readline().strip()
            if not question:
                break
            response = printed = ""

            def show(piece: str, *, complete: bool = False) -> None:
                nonlocal response, printed
                response += piece
                text = _shown(response, thinking=args.enable_thinking, complete=complete)
                if text.startswith(printed):
                    print(text[len(printed) :], end="", flush=True)
                    printed = text
                elif complete:  # never lose the answer to a display mismatch
                    print(f"\n{text}", end="", flush=True)

            try:
                result = session.run(
                    output_dir=args.output_dir / f"turn-{len(history) + 1:03d}",
                    prompt=question,
                    history=history,
                    enable_thinking=args.enable_thinking,
                    max_new_tokens=args.max_new_tokens,
                    overwrite=args.overwrite,
                    on_text=show,
                )
            finally:
                # Also ends the line of a partial answer when generation fails.
                show("", complete=True)
                print(flush=True)
            details = cast(dict[str, object], result["details"])
            history.append((question, cast(str, details["report"])))


def _commit(value: str) -> str:
    if not _COMMIT.fullmatch(value):
        raise argparse.ArgumentTypeError("must be a full 40-character lowercase commit hash")
    return value


def _download(repo: str, revision: str, model_dir: Path) -> None:
    """Download and admit one complete Hub commit before recording its revision.

    A Hub dry run checks per-file commit metadata after downloading: local_dir
    downloads retain old files and can return them when the network fails.
    """
    hub = import_module("huggingface_hub")
    folder = model_dir.expanduser()
    # Remove the record first so an interrupted download never claims the new commit.
    record = folder / _REVISION_FILE
    record.unlink(missing_ok=True)
    hub.snapshot_download(repo, revision=revision, local_dir=folder)
    read_manifest(folder)
    downloaded = cast(
        "list[DryRunFileInfo]",
        hub.snapshot_download(repo, revision=revision, local_dir=folder, dry_run=True),
    )
    inventory = {item.filename: item for item in downloaded}
    for name in (*RUNTIME_FILES, MANIFEST, *(f"mlx/{shard}" for shard in SHARDS)):
        item = inventory.get(name)
        if item is None or item.commit_hash != revision or item.will_download:
            raise InvalidInputError(f"Hub has not confirmed {name} at the requested revision")
    write_private_text(record, revision + "\n", overwrite=True)
