import tempfile
import unittest
from pathlib import Path

from podwatch.config import ConfigError, load_config


def write(text: str) -> Path:
    handle = tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False, encoding="utf-8")
    handle.write(text)
    handle.close()
    return Path(handle.name)


class ConfigTest(unittest.TestCase):
    def test_defaults_and_overrides(self):
        path = write('root = "/data"\n[[pod]]\nname = "a"\nhost = "203.0.113.1"\n'
                     '[[pod]]\nname = "b"\nhost = "203.0.113.2"\nport = 2222\nroot = "/other"\n')
        a, b = load_config(path)
        self.assertEqual((a.port, a.user, a.root), (22, "root", "/data"))
        self.assertEqual((b.port, b.root), (2222, "/other"))

    def test_missing_host(self):
        with self.assertRaises(ConfigError):
            load_config(write('[[pod]]\nname = "a"\n'))

    def test_option_like_host_is_refused(self):
        with self.assertRaises(ConfigError):
            load_config(write('[[pod]]\nname = "a"\nhost = "-oProxyCommand=x"\n'))

    def test_duplicate_names(self):
        with self.assertRaises(ConfigError):
            load_config(write('[[pod]]\nname = "a"\nhost = "h1"\n[[pod]]\nname = "a"\nhost = "h2"\n'))

    def test_unknown_key(self):
        with self.assertRaises(ConfigError):
            load_config(write('[[pod]]\nname = "a"\nhost = "h"\ncolour = "red"\n'))

    def test_empty_file(self):
        with self.assertRaises(ConfigError):
            load_config(write("root = '/x'\n"))


if __name__ == "__main__":
    unittest.main()
