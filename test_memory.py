import json
import tempfile
import unittest
from pathlib import Path

from omni_agent.memory import MemoryError, MemoryStore


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "private" / "memory.json"

    def test_add_list_context_and_delete(self):
        store = MemoryStore(self.path)
        note = store.add("Prefers concise summaries")
        self.assertEqual(store.list_notes()[0]["text"], "Prefers concise summaries")
        self.assertIn("Prefers concise summaries", store.context())
        reloaded = MemoryStore(self.path)
        self.assertEqual(reloaded.list_notes()[0]["id"], note["id"])
        self.assertTrue(reloaded.delete(note["id"]))
        self.assertFalse(reloaded.delete(note["id"]))
        self.assertEqual(MemoryStore(self.path).list_notes(), [])

    def test_invalid_file_is_not_silently_overwritten(self):
        self.path.parent.mkdir(parents=True)
        self.path.write_text("not json", encoding="utf-8")
        with self.assertRaises(MemoryError):
            MemoryStore(self.path)
        self.assertEqual(self.path.read_text(encoding="utf-8"), "not json")

    def test_note_size_and_empty_content_are_rejected(self):
        store = MemoryStore(self.path)
        for value in ("  ", "x" * (store.MAX_NOTE_CHARS + 1)):
            with self.subTest(length=len(value)):
                with self.assertRaises(MemoryError):
                    store.add(value)


if __name__ == "__main__":
    unittest.main()
