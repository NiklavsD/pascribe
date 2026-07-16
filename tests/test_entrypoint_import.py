import importlib
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


class EntrypointImportTests(unittest.TestCase):
    def test_desktop_entrypoint_imports_without_starting_hardware(self):
        fake_image_type = type("FakeImage", (), {})
        pil = types.ModuleType("PIL")
        pil.Image = types.SimpleNamespace(Image=fake_image_type)
        pil.ImageDraw = types.SimpleNamespace()

        pystray = types.ModuleType("pystray")
        pystray.Icon = type("Icon", (), {})
        pystray.MenuItem = type("MenuItem", (), {})
        pystray.Menu = type("Menu", (), {"SEPARATOR": object()})

        stubs = {
            "sounddevice": types.ModuleType("sounddevice"),
            "keyboard": types.ModuleType("keyboard"),
            "pyperclip": types.ModuleType("pyperclip"),
            "PIL": pil,
            "pystray": pystray,
        }

        with tempfile.TemporaryDirectory() as data_dir:
            with patch.dict(sys.modules, stubs), patch.dict(
                os.environ,
                {"PASCRIBE_DATA_DIR": data_dir},
            ):
                sys.modules.pop("pascribe", None)
                module = importlib.import_module("pascribe")
                self.assertEqual(module.__version__, "0.6.0")
                self.assertEqual(module.APP_PATHS.data_dir, module.Path(data_dir).resolve())
                sys.modules.pop("pascribe", None)


if __name__ == "__main__":
    unittest.main()
