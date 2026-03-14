from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from session_resume import ResumePaths
from session_resume import auto_snapshot
from session_resume import has_meaningful_changes
from session_resume import latest_session_file
from session_resume import render_resume
from session_resume import save_snapshot


class TestSessionResume(unittest.TestCase):
    def test_resume_uses_latest_snapshot_when_available(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            continuation = root / "continuation.md"
            sessions = root / "sessions"
            sessions.mkdir()
            continuation.write_text("fallback", encoding="utf-8")
            (sessions / "SESSION_20260314_100000.md").write_text("old", encoding="utf-8")
            (sessions / "SESSION_20260314_110000.md").write_text("new", encoding="utf-8")
            paths = ResumePaths(root=root, continuation_file=continuation, sessions_dir=sessions)

            rendered = render_resume(paths)

            self.assertEqual(rendered, "new")
            self.assertEqual(latest_session_file(paths), sessions / "SESSION_20260314_110000.md")

    def test_resume_falls_back_to_continuation_when_no_snapshot_exists(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            continuation = root / "continuation.md"
            sessions = root / "sessions"
            continuation.write_text("ligne 1\nligne 2", encoding="utf-8")
            paths = ResumePaths(root=root, continuation_file=continuation, sessions_dir=sessions)

            rendered = render_resume(paths)

            self.assertIn("# Resume local", rendered)
            self.assertIn("ligne 1", rendered)
            self.assertIn("## Etat du depot", rendered)

    def test_snapshot_creates_markdown_file_with_required_sections(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            continuation = root / "continuation.md"
            sessions = root / "sessions"
            continuation.write_text("dernier contexte", encoding="utf-8")
            paths = ResumePaths(root=root, continuation_file=continuation, sessions_dir=sessions)

            saved = save_snapshot(
                paths,
                title="Notebook graph",
                summary="Prototype du resume local",
                next_step="Tester resume --last",
                notes="Verifier l'usage dans VSCode.",
            )

            content = saved.read_text(encoding="utf-8")
            self.assertTrue(saved.exists())
            self.assertIn("## Titre", content)
            self.assertIn("Notebook graph", content)
            self.assertIn("## Prochaine etape", content)
            self.assertIn("## Continuation", content)
            self.assertIn("dernier contexte", content)

    def test_has_meaningful_changes_is_false_without_git_repo(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            continuation = root / "continuation.md"
            continuation.write_text("ctx", encoding="utf-8")
            paths = ResumePaths(root=root, continuation_file=continuation, sessions_dir=root / "sessions")

            self.assertFalse(has_meaningful_changes(paths))

    def test_auto_snapshot_skips_when_no_changes(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            continuation = root / "continuation.md"
            continuation.write_text("ctx", encoding="utf-8")
            paths = ResumePaths(root=root, continuation_file=continuation, sessions_dir=root / "sessions")

            saved = auto_snapshot(paths, min_interval_seconds=0)

            self.assertIsNone(saved)


if __name__ == "__main__":
    unittest.main()
