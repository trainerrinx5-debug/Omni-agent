import json
import tempfile
import unittest
from pathlib import Path

from omni_agent.safety import WriteApproval
from omni_agent.tools import ToolError, ToolRegistry, WorkspaceTools, safe_calculate


class CalculatorTests(unittest.TestCase):
    def test_basic_arithmetic(self):
        self.assertEqual(safe_calculate("(19.5 * 3) / 2"), 29.25)
        self.assertEqual(safe_calculate("2 ** 8"), 256)
        self.assertEqual(safe_calculate("-7 // 2"), -4)

    def test_disallows_code_and_unbounded_math(self):
        for expression in ("__import__('os')", "2 + True", "2 ** 100", "1 / 0"):
            with self.subTest(expression=expression):
                with self.assertRaises(ToolError):
                    safe_calculate(expression)


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "workspace"
        self.root.mkdir()

    def test_path_traversal_and_symlink_escape_are_rejected(self):
        outside = self.base / "outside.txt"
        outside.write_text("secret", encoding="utf-8")
        tools = WorkspaceTools(self.root, WriteApproval(auto_approve=True))
        with self.assertRaises(ToolError):
            tools.read_file({"path": "../outside.txt"})
        link = self.root / "escape.txt"
        try:
            link.symlink_to(outside)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks are unavailable in this environment")
        with self.assertRaises(ToolError):
            tools.read_file({"path": "escape.txt"})

    def test_write_requires_approval_and_stays_in_workspace(self):
        deny = WorkspaceTools(self.root, WriteApproval(confirm_fn=lambda _path, _exists: False))
        registry = ToolRegistry(deny.specs())
        result = registry.execute("workspace_write", {"path": "notes.txt", "content": "hello"})
        self.assertIn("write was denied", result)
        self.assertFalse((self.root / "notes.txt").exists())

        approve = WorkspaceTools(self.root, WriteApproval(confirm_fn=lambda _path, _exists: True))
        written = json.loads(approve.write_file({"path": "nested/notes.txt", "content": "hello"}))
        self.assertTrue(written["written"])
        self.assertEqual((self.root / "nested/notes.txt").read_text(encoding="utf-8"), "hello")

    def test_read_and_list(self):
        (self.root / "a.txt").write_text("hello", encoding="utf-8")
        tools = WorkspaceTools(self.root, WriteApproval())
        read = json.loads(tools.read_file({"path": "a.txt"}))
        self.assertEqual(read["content"], "hello")
        listing = json.loads(tools.list_files({}))
        self.assertEqual(listing["entries"][0]["name"], "a.txt")


if __name__ == "__main__":
    unittest.main()
