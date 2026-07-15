#!/usr/bin/env python3
"""Validate committed AngioStress benchmark artifacts.

The default/metadata-only mode is intentionally offline: it verifies the
committed root MANIFEST.json, file sizes, SHA-256 digests, and JSON syntax
without requiring external datasets or model checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


@dataclass
class ValidationResult:
    checks: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def check(self, condition: bool, message: str) -> None:
        self.checks += 1
        if not condition:
            self.errors.append(message)

    def passed(self, message: str) -> None:
        self.checks += 1

    def warn(self, message: str) -> None:
        self.warnings.append(message)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_manifest_path(path_text: str) -> bool:
    path = Path(path_text)
    return (
        "\x00" not in path_text
        and not any(ord(char) < 32 for char in path_text)
        and not path.is_absolute()
        and ".." not in path.parts
        and path_text not in {"", "."}
    )


def valid_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(char in "0123456789abcdefABCDEF" for char in value)


def load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def tracked_files(root: Path) -> list[Path]:
    """Return git-tracked files when possible, otherwise all non-.git files."""
    try:
        completed = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        candidates = [root / p.decode() for p in completed.stdout.split(b"\0") if p]
    except (OSError, subprocess.CalledProcessError):
        candidates = [p for p in root.rglob("*") if ".git" not in p.parts]
    return [p for p in candidates if p.is_file() and not p.is_symlink()]


def validate_root_manifest(root: Path, result: ValidationResult) -> None:
    manifest_path = root / "MANIFEST.json"
    result.check(manifest_path.is_file(), "missing root MANIFEST.json")
    if not manifest_path.is_file():
        return

    try:
        manifest = load_json(manifest_path)
    except json.JSONDecodeError as exc:
        result.check(False, f"MANIFEST.json is not valid JSON: {exc}")
        return

    result.check(isinstance(manifest, dict), "MANIFEST.json must contain a JSON object")
    if not isinstance(manifest, dict):
        return

    files = manifest.get("files")
    result.check(isinstance(files, list), "MANIFEST.json must contain a files array")
    if not isinstance(files, list):
        return

    file_count = manifest.get("file_count")
    result.check(file_count == len(files), f"MANIFEST.json file_count={file_count!r} but files has {len(files)} entries")

    seen: set[str] = set()
    for index, entry in enumerate(files):
        prefix = f"MANIFEST.json files[{index}]"
        result.check(isinstance(entry, dict), f"{prefix} must be an object")
        if not isinstance(entry, dict):
            continue

        rel_path = entry.get("path")
        expected_size = entry.get("size_bytes")
        expected_sha256 = entry.get("sha256")
        result.check(isinstance(rel_path, str) and safe_manifest_path(rel_path), f"{prefix} has unsafe or missing path: {rel_path!r}")
        result.check(isinstance(expected_size, int) and expected_size >= 0, f"{prefix} has invalid size_bytes: {expected_size!r}")
        result.check(valid_sha256(expected_sha256), f"{prefix} has invalid sha256: {expected_sha256!r}")
        if not (isinstance(rel_path, str) and safe_manifest_path(rel_path)):
            continue

        result.check(rel_path not in seen, f"duplicate manifest path: {rel_path}")
        seen.add(rel_path)

        artifact_path = root / rel_path
        result.check(artifact_path.is_file() and not artifact_path.is_symlink(), f"manifested file is missing or is a symlink: {rel_path}")
        if not artifact_path.is_file() or artifact_path.is_symlink():
            continue

        try:
            actual_size = artifact_path.stat().st_size
        except OSError as exc:
            result.check(False, f"cannot stat file {rel_path}: {exc}")
            continue
        result.check(actual_size == expected_size, f"size mismatch for {rel_path}: expected {expected_size}, got {actual_size}")

        if valid_sha256(expected_sha256):
            try:
                actual_sha256 = sha256_file(artifact_path)
            except OSError as exc:
                result.check(False, f"cannot read file {rel_path}: {exc}")
                continue
            result.check(actual_sha256 == expected_sha256, f"sha256 mismatch for {rel_path}: expected {expected_sha256}, got {actual_sha256}")


def validate_json_syntax(root: Path, result: ValidationResult) -> None:
    json_files = [p for p in tracked_files(root) if p.suffix == ".json"]
    result.check(bool(json_files), "no tracked JSON files found to validate")
    for path in json_files:
        rel = path.relative_to(root).as_posix()
        try:
            load_json(path)
        except json.JSONDecodeError as exc:
            result.check(False, f"invalid JSON in {rel}: {exc}")
        else:
            result.passed(f"valid JSON: {rel}")


def validate_smoke_structure(root: Path, result: ValidationResult) -> None:
    experiment_root = root / "experiments"
    experiment_dirs = [p.parent for p in experiment_root.rglob("run_*.py") if p.is_file()] if experiment_root.is_dir() else []
    result.check(bool(experiment_dirs), "no runnable experiment scripts found under experiments/")
    for exp_dir in sorted(set(experiment_dirs)):
        rel = exp_dir.relative_to(root).as_posix()
        has_run_doc = (exp_dir / "RUN.md").is_file()
        has_validation_doc = (exp_dir / "VALIDATION.md").is_file()
        has_outputs = any(
            child.is_dir() and not child.is_symlink() and child.name.lower().startswith("outputs")
            for child in exp_dir.iterdir()
        )
        result.check(has_run_doc or has_validation_doc or has_outputs, f"experiment {rel} lacks RUN.md, VALIDATION.md, or outputs*/ evidence")

    root_docs = ["README.md", "RUN_BENCHMARK.md", "DATA_SOURCES.md", "CITATION.cff"]
    for doc in root_docs:
        result.check((root / doc).is_file(), f"missing top-level documentation file: {doc}")


def run_validation(root: Path, mode: str) -> ValidationResult:
    result = ValidationResult()
    validate_root_manifest(root, result)
    validate_json_syntax(root, result)

    if mode in {"smoke", "full"}:
        validate_smoke_structure(root, result)

    if mode == "full":
        result.warn("--full currently performs all offline checks; external dataset/checkpoint reruns remain documented per experiment.")

    return result


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate committed AngioStress benchmark artifacts.")
    parser.add_argument(
        "--mode",
        choices=("metadata-only", "smoke", "full"),
        default="metadata-only",
        help="validation depth; all modes are offline unless future checks add external reruns",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="shortcut for --mode metadata-only",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="repository root to validate (default: directory containing this script)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    mode = "metadata-only" if args.metadata_only else args.mode
    root = args.root.resolve()

    result = run_validation(root, mode)

    if result.ok:
        print(f"artifact validation passed: mode={mode}, checks={result.checks}, root={root}")
    else:
        print(f"artifact validation failed: mode={mode}, checks={result.checks}, errors={len(result.errors)}, root={root}", file=sys.stderr)
        for error in result.errors:
            print(f"ERROR: {error}", file=sys.stderr)

    for warning in result.warnings:
        print(f"WARNING: {warning}", file=sys.stderr)

    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
