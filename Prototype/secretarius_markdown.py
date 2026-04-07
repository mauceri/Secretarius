from __future__ import annotations

import re
from typing import Any


SECRETARIUS_BLOCK_RE = re.compile(r"```secretarius[ \t]*\n(.*?)```", re.DOTALL | re.IGNORECASE)


class SecretariusMarkdownError(ValueError):
    pass


def _parse_key_value_lines(block: str) -> dict[str, str]:
    args: dict[str, str] = {}
    for raw_line in (block or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise SecretariusMarkdownError(f"Invalid DSL line: {raw_line}")
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise SecretariusMarkdownError(f"Invalid DSL key: {raw_line}")
        args[key] = value
    return args


def parse_secretarius_markdown(markdown_text: str) -> dict[str, Any] | None:
    text = markdown_text or ""
    matches = list(SECRETARIUS_BLOCK_RE.finditer(text))
    if not matches:
        return None
    if len(matches) > 1:
        raise SecretariusMarkdownError("Multiple secretarius blocks are not supported.")

    match = matches[0]
    block = match.group(1)
    args = _parse_key_value_lines(block)
    action = (args.get("action") or "").strip().lower()
    if not action:
        raise SecretariusMarkdownError("Missing required key: action")

    content = (text[:match.start()] + text[match.end():]).strip()
    content = re.sub(r"\n{3,}", "\n\n", content)
    return {
        "action": action,
        "args": args,
        "content": content,
        "raw_block": block,
    }


def _normalize_tags(tags_value: str) -> str:
    tags = []
    for part in (tags_value or "").split(","):
        value = part.strip()
        if not value:
            continue
        if not value.startswith("#"):
            value = f"#{value}"
        tags.append(value)
    return " ".join(tags)


def _build_document_text(args: dict[str, str], content: str, require_doc_id: bool) -> str:
    lines: list[str] = []

    doc_id = (args.get("doc_id") or "").strip()
    if require_doc_id and not doc_id:
        raise SecretariusMarkdownError("Missing required key: doc_id")
    if doc_id:
        lines.append(f"doc_id: {doc_id}")

    type_note = (args.get("type_note") or "").strip()
    if type_note:
        lines.append(f"type_note: {type_note}")

    title = (args.get("title") or "").strip()
    if title:
        lines.append(f"title: {title}")

    tags = _normalize_tags(args.get("tags") or "")
    if tags:
        lines.append(tags)

    body = (content or "").strip()
    if not body:
        raise SecretariusMarkdownError("Missing document content outside the secretarius block.")
    lines.append(body)
    return "\n".join(lines)


def render_secretarius_command(parsed: dict[str, Any]) -> str:
    action = str(parsed.get("action") or "").strip().lower()
    args = parsed.get("args")
    content = str(parsed.get("content") or "").strip()
    if not isinstance(args, dict):
        raise SecretariusMarkdownError("Invalid parsed DSL arguments.")

    if action == "index":
        return "/index\n" + _build_document_text(args, content, require_doc_id=False)

    if action == "update":
        return "/update\n" + _build_document_text(args, content, require_doc_id=True)

    if action == "req":
        query = (args.get("query") or "").strip()
        if not query:
            raise SecretariusMarkdownError("Missing required key: query")
        if content:
            raise SecretariusMarkdownError("Action 'req' does not accept document content outside the block.")
        return f"/req {query}"

    if action == "exp":
        text = (args.get("text") or "").strip() or content
        if not text.strip():
            raise SecretariusMarkdownError("Action 'exp' requires text or document content.")
        return "/exp " + text.strip()

    raise SecretariusMarkdownError(f"Unsupported action: {action}")
