import json
import io
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import validate_artifacts


class ValidateArtifactsTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def write_manifest(self, files):
        (self.tmpdir / "MANIFEST.json").write_text(
            json.dumps(
                {
                    "schema_version": 3,
                    "package": "test-artifacts",
                    "file_count": len(files),
                    "files": files,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def test_metadata_only_accepts_matching_manifest_and_json(self):
        payload = self.tmpdir / "artifact.json"
        payload.write_text('{"ok": true}\n', encoding="utf-8")
        self.write_manifest(
            [
                {
                    "path": "artifact.json",
                    "size_bytes": payload.stat().st_size,
                    "sha256": validate_artifacts.sha256_file(payload),
                }
            ]
        )

        result = validate_artifacts.run_validation(self.tmpdir, "metadata-only")

        self.assertTrue(result.ok, result.errors)
        self.assertGreater(result.checks, 0)

    def test_metadata_only_rejects_hash_mismatch(self):
        payload = self.tmpdir / "artifact.json"
        payload.write_text('{"ok": true}\n', encoding="utf-8")
        self.write_manifest(
            [
                {
                    "path": "artifact.json",
                    "size_bytes": payload.stat().st_size,
                    "sha256": "0" * 64,
                }
            ]
        )

        result = validate_artifacts.run_validation(self.tmpdir, "metadata-only")

        self.assertFalse(result.ok)
        self.assertTrue(any("sha256 mismatch" in error for error in result.errors), result.errors)

    def test_metadata_only_rejects_unsupported_schema_version(self):
        self.write_manifest([])
        manifest_path = self.tmpdir / "MANIFEST.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["schema_version"] = 2
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        result = validate_artifacts.run_validation(self.tmpdir, "metadata-only")

        self.assertFalse(result.ok)
        self.assertTrue(any("unsupported schema_version" in error for error in result.errors), result.errors)

    def test_metadata_only_rejects_non_integer_file_count(self):
        self.write_manifest([])
        manifest_path = self.tmpdir / "MANIFEST.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["file_count"] = "0"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        result = validate_artifacts.run_validation(self.tmpdir, "metadata-only")

        self.assertFalse(result.ok)
        self.assertTrue(any("file_count='0'" in error for error in result.errors), result.errors)

    def test_metadata_only_rejects_unsafe_manifest_path(self):
        self.write_manifest(
            [
                {
                    "path": "../escape.json",
                    "size_bytes": 0,
                    "sha256": "0" * 64,
                }
            ]
        )

        result = validate_artifacts.run_validation(self.tmpdir, "metadata-only")

        self.assertFalse(result.ok)
        self.assertTrue(any("unsafe" in error for error in result.errors), result.errors)

    def test_metadata_only_rejects_manifest_path_with_null_byte(self):
        self.write_manifest(
            [
                {
                    "path": "bad\u0000name.json",
                    "size_bytes": 0,
                    "sha256": "0" * 64,
                }
            ]
        )

        result = validate_artifacts.run_validation(self.tmpdir, "metadata-only")

        self.assertFalse(result.ok)
        self.assertTrue(any("unsafe" in error for error in result.errors), result.errors)

    def test_metadata_only_rejects_manifest_path_with_del_character(self):
        self.write_manifest(
            [
                {
                    "path": "bad\u007fname.json",
                    "size_bytes": 0,
                    "sha256": "0" * 64,
                }
            ]
        )

        result = validate_artifacts.run_validation(self.tmpdir, "metadata-only")

        self.assertFalse(result.ok)
        self.assertTrue(any("unsafe" in error for error in result.errors), result.errors)

    def test_metadata_only_reports_missing_manifested_file(self):
        self.write_manifest(
            [
                {
                    "path": "missing.json",
                    "size_bytes": 12,
                    "sha256": "0" * 64,
                }
            ]
        )

        result = validate_artifacts.run_validation(self.tmpdir, "metadata-only")

        self.assertFalse(result.ok)
        self.assertTrue(any("manifested file is missing" in error for error in result.errors), result.errors)

    def test_metadata_only_reports_size_mismatch(self):
        payload = self.tmpdir / "artifact.json"
        payload.write_text('{"ok": true}\n', encoding="utf-8")
        self.write_manifest(
            [
                {
                    "path": "artifact.json",
                    "size_bytes": payload.stat().st_size + 1,
                    "sha256": validate_artifacts.sha256_file(payload),
                }
            ]
        )

        result = validate_artifacts.run_validation(self.tmpdir, "metadata-only")

        self.assertFalse(result.ok)
        self.assertTrue(any("size mismatch" in error for error in result.errors), result.errors)

    def test_metadata_only_rejects_symlink_manifest_entry(self):
        target = self.tmpdir / "outside.json"
        target.write_text('{"secret": true}\n', encoding="utf-8")
        symlink = self.tmpdir / "artifact.json"
        symlink.symlink_to(target)
        self.write_manifest(
            [
                {
                    "path": "artifact.json",
                    "size_bytes": target.stat().st_size,
                    "sha256": validate_artifacts.sha256_file(target),
                }
            ]
        )

        result = validate_artifacts.run_validation(self.tmpdir, "metadata-only")

        self.assertFalse(result.ok)
        self.assertTrue(any("symlink" in error for error in result.errors), result.errors)

    def test_metadata_only_reports_file_read_errors_without_crashing(self):
        payload = self.tmpdir / "artifact.json"
        payload.write_text('{"ok": true}\n', encoding="utf-8")
        self.write_manifest(
            [
                {
                    "path": "artifact.json",
                    "size_bytes": payload.stat().st_size,
                    "sha256": "0" * 64,
                }
            ]
        )

        with mock.patch("validate_artifacts.sha256_file", side_effect=OSError("permission denied")):
            result = validate_artifacts.run_validation(self.tmpdir, "metadata-only")

        self.assertFalse(result.ok)
        self.assertTrue(any("cannot read file artifact.json" in error for error in result.errors), result.errors)

    def test_tracked_files_fallback_skips_symlinks(self):
        real = self.tmpdir / "artifact.json"
        real.write_text("{}\n", encoding="utf-8")
        link = self.tmpdir / "linked.json"
        link.symlink_to(real)

        with mock.patch("validate_artifacts.subprocess.run", side_effect=OSError("git unavailable")):
            paths = validate_artifacts.tracked_files(self.tmpdir)

        self.assertIn(real, paths)
        self.assertNotIn(link, paths)

    def test_tracked_files_falls_back_when_git_times_out(self):
        artifact = self.tmpdir / "artifact.json"
        artifact.write_text("{}\n", encoding="utf-8")

        with mock.patch(
            "validate_artifacts.subprocess.run",
            side_effect=validate_artifacts.subprocess.TimeoutExpired("git", 30),
        ):
            paths = validate_artifacts.tracked_files(self.tmpdir)

        self.assertIn(artifact, paths)

    def test_tracked_files_preserves_non_utf8_git_paths(self):
        raw_path = b"artifact-\xff.json"
        completed = mock.Mock(stdout=raw_path + b"\0")

        with (
            mock.patch("validate_artifacts.subprocess.run", return_value=completed),
            mock.patch("pathlib.Path.is_file", return_value=True),
            mock.patch("pathlib.Path.is_symlink", return_value=False),
        ):
            paths = validate_artifacts.tracked_files(self.tmpdir)

        self.assertEqual(paths, [self.tmpdir / os.fsdecode(raw_path)])

    def test_smoke_structure_rejects_symlink_outputs_evidence(self):
        exp_dir = self.tmpdir / "experiments" / "main" / "run-symlink-output"
        exp_dir.mkdir(parents=True)
        (exp_dir / "run_demo.py").write_text("print('ok')\n", encoding="utf-8")
        (exp_dir / "RUN.md").write_text("# Run\n", encoding="utf-8")
        external_outputs = self.tmpdir / "external-outputs"
        external_outputs.mkdir()
        (exp_dir / "outputs_evil").symlink_to(external_outputs, target_is_directory=True)
        for doc in ["README.md", "RUN_BENCHMARK.md", "DATA_SOURCES.md", "CITATION.cff"]:
            (self.tmpdir / doc).write_text("placeholder\n", encoding="utf-8")
        result = validate_artifacts.ValidationResult()

        validate_artifacts.validate_smoke_structure(self.tmpdir, result)

        self.assertFalse(result.ok)
        self.assertTrue(any("symlink named outputs" in error for error in result.errors), result.errors)

    def test_smoke_structure_accepts_case_insensitive_outputs_directory(self):
        exp_dir = self.tmpdir / "experiments" / "main" / "run-uppercase-output"
        exp_dir.mkdir(parents=True)
        (exp_dir / "run_demo.py").write_text("print('ok')\n", encoding="utf-8")
        (exp_dir / "Outputs").mkdir()
        for doc in ["README.md", "RUN_BENCHMARK.md", "DATA_SOURCES.md", "CITATION.cff"]:
            (self.tmpdir / doc).write_text("placeholder\n", encoding="utf-8")
        result = validate_artifacts.ValidationResult()

        validate_artifacts.validate_smoke_structure(self.tmpdir, result)

        self.assertTrue(result.ok, result.errors)

    def test_smoke_structure_finds_run_scripts_directly_under_experiments(self):
        exp_dir = self.tmpdir / "experiments"
        exp_dir.mkdir()
        (exp_dir / "run_root.py").write_text("print('ok')\n", encoding="utf-8")
        for doc in ["README.md", "RUN_BENCHMARK.md", "DATA_SOURCES.md", "CITATION.cff"]:
            (self.tmpdir / doc).write_text("placeholder\n", encoding="utf-8")
        result = validate_artifacts.ValidationResult()

        validate_artifacts.validate_smoke_structure(self.tmpdir, result)

        self.assertFalse(result.ok)
        self.assertTrue(any("experiment experiments lacks" in error for error in result.errors), result.errors)

    def test_metadata_only_rejects_non_hex_sha256(self):
        payload = self.tmpdir / "artifact.json"
        payload.write_text('{"ok": true}\n', encoding="utf-8")
        self.write_manifest(
            [
                {
                    "path": "artifact.json",
                    "size_bytes": payload.stat().st_size,
                    "sha256": "z" * 64,
                }
            ]
        )

        result = validate_artifacts.run_validation(self.tmpdir, "metadata-only")

        self.assertFalse(result.ok)
        self.assertTrue(any("invalid sha256" in error for error in result.errors), result.errors)

    def test_cli_rejects_conflicting_mode_flags(self):
        with self.assertRaises(SystemExit):
            validate_artifacts.parse_args(["--mode", "smoke", "--metadata-only"])

    def test_main_rejects_missing_root(self):
        missing_root = self.tmpdir / "missing"
        stderr = io.StringIO()

        with mock.patch("sys.stderr", stderr):
            exit_code = validate_artifacts.main(["--root", str(missing_root)])

        self.assertEqual(exit_code, 1)
        self.assertIn("root is not a directory", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
