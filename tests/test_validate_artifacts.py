import json
import shutil
import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
