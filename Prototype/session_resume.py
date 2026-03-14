from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import subprocess
from typing import Iterable


DEFAULT_SESSIONS_DIRNAME = "sessions"


@dataclass(frozen=True)
class ResumePaths:
    root: Path
    continuation_file: Path
    sessions_dir: Path


def build_paths(root: Path | None = None) -> ResumePaths:
    project_root = (root or Path(__file__).resolve().parent).resolve()
    return ResumePaths(
        root=project_root,
        continuation_file=project_root / "continuation.md",
        sessions_dir=project_root / DEFAULT_SESSIONS_DIRNAME,
    )


def list_session_files(paths: ResumePaths) -> list[Path]:
    if not paths.sessions_dir.exists():
        return []
    return sorted(
        (path for path in paths.sessions_dir.glob("SESSION_*.md") if path.is_file()),
        key=lambda path: path.name,
    )


def latest_session_file(paths: ResumePaths) -> Path | None:
    sessions = list_session_files(paths)
    if not sessions:
        return None
    return sessions[-1]


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _run_git_command(args: Iterable[str], cwd: Path) -> str:
    completed = subprocess.run(
        list(args),
        cwd=str(cwd),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


def _current_branch(paths: ResumePaths) -> str:
    branch = _run_git_command(["git", "branch", "--show-current"], cwd=paths.root)
    return branch or "-"


def _git_status_lines(paths: ResumePaths) -> list[str]:
    status = _run_git_command(["git", "status", "--short"], cwd=paths.root)
    if not status:
        return ["clean"]
    return status.splitlines()


def _recent_commit(paths: ResumePaths) -> str:
    commit = _run_git_command(["git", "log", "-1", "--pretty=format:%h %cs %s"], cwd=paths.root)
    return commit or "-"


def _continuation_excerpt(paths: ResumePaths, max_lines: int = 20) -> str:
    if not paths.continuation_file.exists():
        return "continuation.md absent."

    lines = _read_text(paths.continuation_file).splitlines()
    excerpt = lines[-max_lines:]
    return "\n".join(excerpt).strip() or "continuation.md vide."


def render_resume(paths: ResumePaths) -> str:
    latest = latest_session_file(paths)
    if latest is not None:
        return _read_text(latest)

    return "\n".join(
        [
            "# Resume local",
            "",
            "Aucun snapshot de session n'est disponible.",
            "",
            "## Contexte de secours",
            "",
            f"- Branche git: `{_current_branch(paths)}`",
            f"- Dernier commit: `{_recent_commit(paths)}`",
            f"- Fichier source: `{paths.continuation_file.name}`",
            "",
            "## Etat du depot",
            "",
            *[f"- {line}" for line in _git_status_lines(paths)],
            "",
            "## Extrait de continuation",
            "",
            _continuation_excerpt(paths),
        ]
    ).strip()


def has_meaningful_changes(paths: ResumePaths) -> bool:
    status = _run_git_command(["git", "status", "--short", "--", "Prototype"], cwd=paths.root.parent)
    return bool(status.strip())


def latest_snapshot_age_seconds(paths: ResumePaths, now: datetime | None = None) -> float | None:
    latest = latest_session_file(paths)
    if latest is None:
        return None
    current = now or datetime.now()
    return max(0.0, current.timestamp() - latest.stat().st_mtime)


def save_snapshot(
    paths: ResumePaths,
    *,
    title: str,
    summary: str,
    next_step: str,
    notes: str = "",
) -> Path:
    paths.sessions_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = paths.sessions_dir / f"SESSION_{timestamp}.md"

    content_lines = [
        f"# Session Snapshot - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Titre",
        "",
        title.strip() or "(sans titre)",
        "",
        "## Resume",
        "",
        summary.strip() or "(sans resume)",
        "",
        "## Prochaine etape",
        "",
        next_step.strip() or "(non renseignee)",
        "",
        "## Contexte depot",
        "",
        f"- Branche git: `{_current_branch(paths)}`",
        f"- Dernier commit: `{_recent_commit(paths)}`",
        "",
        "## Etat du depot",
        "",
        *[f"- {line}" for line in _git_status_lines(paths)],
        "",
        "## Continuation",
        "",
        _continuation_excerpt(paths),
    ]

    cleaned_notes = notes.strip()
    if cleaned_notes:
        content_lines.extend(["", "## Notes", "", cleaned_notes])

    target.write_text("\n".join(content_lines).strip() + "\n", encoding="utf-8")
    return target


def auto_snapshot(
    paths: ResumePaths,
    *,
    title: str = "Auto snapshot",
    summary: str = "Snapshot periodique du contexte local Prototype.",
    next_step: str = "Relire resume --last avant reprise.",
    notes: str = "",
    min_interval_seconds: int = 1800,
) -> Path | None:
    age_seconds = latest_snapshot_age_seconds(paths)
    if age_seconds is not None and age_seconds < float(min_interval_seconds):
        return None

    if not has_meaningful_changes(paths):
        return None

    return save_snapshot(
        paths,
        title=title,
        summary=summary,
        next_step=next_step,
        notes=notes,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local session snapshot and resume helper for Prototype.")
    parser.add_argument(
        "--root",
        default=None,
        help="Project root containing continuation.md and optional sessions/.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    resume_parser = subparsers.add_parser("resume", help="Print the latest saved session or a fallback resume.")
    resume_parser.add_argument(
        "--last",
        action="store_true",
        help="Accepted for compatibility; resumes the latest snapshot.",
    )

    save_parser = subparsers.add_parser("snapshot", help="Save a session snapshot as Markdown.")
    save_parser.add_argument("--title", required=True, help="Short title for the snapshot.")
    save_parser.add_argument("--summary", required=True, help="What was done in this session.")
    save_parser.add_argument("--next-step", required=True, help="First recommended next action.")
    save_parser.add_argument("--notes", default="", help="Optional additional notes.")

    auto_parser = subparsers.add_parser("snapshot-auto", help="Save a snapshot only if changes justify it.")
    auto_parser.add_argument("--title", default="Auto snapshot", help="Short title for the snapshot.")
    auto_parser.add_argument(
        "--summary",
        default="Snapshot periodique du contexte local Prototype.",
        help="Summary used for the auto snapshot.",
    )
    auto_parser.add_argument(
        "--next-step",
        default="Relire resume --last avant reprise.",
        help="Recommended next action stored in the auto snapshot.",
    )
    auto_parser.add_argument("--notes", default="", help="Optional additional notes.")
    auto_parser.add_argument(
        "--min-interval-seconds",
        type=int,
        default=1800,
        help="Do not write a new snapshot if the latest one is newer than this interval.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    paths = build_paths(Path(args.root) if args.root else None)

    if args.command == "resume":
        print(render_resume(paths))
        return 0

    if args.command == "snapshot":
        saved_path = save_snapshot(
            paths,
            title=args.title,
            summary=args.summary,
            next_step=args.next_step,
            notes=args.notes,
        )
        print(saved_path)
        return 0

    if args.command == "snapshot-auto":
        saved_path = auto_snapshot(
            paths,
            title=args.title,
            summary=args.summary,
            next_step=args.next_step,
            notes=args.notes,
            min_interval_seconds=args.min_interval_seconds,
        )
        if saved_path is None:
            print("SKIPPED")
        else:
            print(saved_path)
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
