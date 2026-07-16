import json
import sys
import tempfile
import unittest
from pathlib import Path

from pascribe_app.paths import build_app_paths, prepare_app_paths, resolve_recording_path
from pascribe_app.secrets import protect_secret, unprotect_secret
from pascribe_app.storage import atomic_write_json, load_json
from pascribe_app.validation import normalize_config, validate_settings


DEFAULTS = {
    "sample_rate": 16000,
    "buffer_minutes": 60,
    "history_max_entries": 100,
    "whisper_model": "large-v3",
    "whisper_device": "cuda",
    "hotkey_prefix": "ctrl+alt",
    "recording_path": "recordings",
    "daily_recording": False,
    "delete_after_transcribe": False,
    "mic_device": None,
    "system_device": None,
    "mic_device_name": None,
    "system_device_name": None,
    "hotkeys": {"9": 1},
    "homelab_url": None,
    "assemblyai_key": "",
}


class StorageAndConfigTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform == "win32", "Windows DPAPI test")
    def test_dpapi_secret_round_trip(self):
        protected = protect_secret("test-api-key")
        self.assertIsNotNone(protected)
        self.assertNotIn("test-api-key", protected)
        self.assertEqual(unprotect_secret(protected), "test-api-key")

    def test_atomic_json_round_trip_and_corrupt_backup(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            atomic_write_json(path, {"hello": "world"})
            self.assertEqual(load_json(path, dict).data, {"hello": "world"})

            path.write_text("{broken", encoding="utf-8")
            recovered = load_json(path, lambda: {"safe": True})
            self.assertTrue(recovered.recovered)
            self.assertEqual(recovered.data, {"safe": True})
            self.assertIsNotNone(recovered.backup_path)
            self.assertTrue(recovered.backup_path.exists())

    def test_config_normalization_and_validation(self):
        config, warnings = normalize_config({
            "buffer_minutes": 999,
            "whisper_device": "magic",
            "homelab_url": "not a url",
            "hotkeys": {},
        }, DEFAULTS)
        self.assertEqual(config["buffer_minutes"], 60)
        self.assertEqual(config["whisper_device"], "cuda")
        self.assertIsNone(config["homelab_url"])
        self.assertEqual(config["hotkeys"], {"9": 1})
        self.assertGreaterEqual(len(warnings), 4)

        errors = validate_settings({
            "buffer_minutes": 0,
            "hotkey_prefix": "",
            "recording_path": "",
            "homelab_url": "ftp://example.com",
            "whisper_model": "unknown",
            "whisper_device": "magic",
        })
        self.assertGreaterEqual(len(errors), 6)

    def test_legacy_paths_migrate_without_moving_source_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            app_dir = root / "app"
            local_dir = root / "local"
            app_dir.mkdir()
            (app_dir / "config.json").write_text(json.dumps({
                "recording_path": "recordings",
            }), encoding="utf-8")
            (app_dir / "history.json").write_text("[]", encoding="utf-8")
            paths = build_app_paths(
                app_dir,
                environ={"LOCALAPPDATA": str(local_dir)},
                platform="win32",
            )
            migrated = prepare_app_paths(paths)
            self.assertEqual(set(migrated), {"config.json", "history.json"})
            migrated_config = json.loads(paths.config.read_text(encoding="utf-8"))
            self.assertEqual(
                Path(migrated_config["recording_path"]),
                (app_dir / "recordings").resolve(),
            )
            self.assertFalse((app_dir / "config.json").exists())
            self.assertTrue((app_dir / "config.json.migrated.bak").exists())
            self.assertEqual(resolve_recording_path("relative", paths), paths.data_dir / "relative")


if __name__ == "__main__":
    unittest.main()
