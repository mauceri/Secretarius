from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest


SECRETARIUS_ROOT = Path(__file__).resolve().parent
DEFAULT_PROMPT_PATH = SECRETARIUS_ROOT / "prompts" / "prompt_extracteur.txt"
CHUNK_DATA_PATH = SECRETARIUS_ROOT / "vendor" / "chunk_data.py"
_CACHED_CHUNKER: Any | None = None

_JSON_CODEBLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_JSON_STRING_RE = re.compile(r'"((?:\\.|[^"\\])*)"')
LOGGER = logging.getLogger("secretarius.extractor")

try:
    from .runtime_paths import resolve_nltk_data_path, resolve_sentence_model_path
except ImportError:
    _RUNTIME_PATHS = SECRETARIUS_ROOT / "runtime_paths.py"
    _RUNTIME_PATHS_SPEC = importlib.util.spec_from_file_location("secretarius_runtime_paths", _RUNTIME_PATHS)
    if _RUNTIME_PATHS_SPEC is None or _RUNTIME_PATHS_SPEC.loader is None:
        raise RuntimeError(f"unable to load runtime_paths from {_RUNTIME_PATHS}")
    _runtime_paths_module = importlib.util.module_from_spec(_RUNTIME_PATHS_SPEC)
    sys.modules[_RUNTIME_PATHS_SPEC.name] = _runtime_paths_module
    _RUNTIME_PATHS_SPEC.loader.exec_module(_runtime_paths_module)
    resolve_nltk_data_path = _runtime_paths_module.resolve_nltk_data_path
    resolve_sentence_model_path = _runtime_paths_module.resolve_sentence_model_path


def extract_expressions(
    text: str,
    *,
    llama_cpp_url: str = "http://127.0.0.1:8989/v1/chat/completions",
    llama_cpp_model: str = "local-llama-cpp",
    timeout_s: float = 120.0,
    max_tokens: int = 512,
    prompt_path: str | None = None,
    seed: int = 42,
    debug_return_raw: bool = False,
    per_chunk_llm: bool = False,
) -> dict[str, Any]:
    cleaned = (text or "").strip()
    if not cleaned:
        return {"chunks": [], "by_chunk": [], "expressions": [], "warning": "empty input text"}

    system_prompt = _load_prompt_text(prompt_path)
    if system_prompt is None:
        return {"chunks": [], "by_chunk": [], "expressions": [], "warning": "unable to load prompt"}

    chunker, chunker_warning = _load_chunker()
    if chunker is None:
        return {"chunks": [], "by_chunk": [], "expressions": [], "warning": chunker_warning}

    try:
        chunks = chunker.chunk(cleaned)  # type: ignore[attr-defined]
    except Exception as exc:
        return {"chunks": [], "by_chunk": [], "expressions": [], "warning": f"chunking failed: {exc}"}
    chunks = [c for c in chunks if isinstance(c, str) and c.strip()]
    if not chunks:
        # Fallback pragmatique: traiter le texte complet en un seul chunk.
        chunks = [cleaned]
        warnings = ["chunker returned no chunks; fallback to single raw-text chunk"]
    else:
        warnings = []
    by_chunk: list[dict[str, Any]] = []
    all_expressions: list[str] = []
    filtered_out_total = 0

    if per_chunk_llm:
        source_texts = chunks
        warn_prefix = "chunk"
    else:
        source_texts = [cleaned]
        warn_prefix = "global"

    extracted_by_source: list[tuple[list[str], str | None, int, str]] = []
    for idx, source_text in enumerate(source_texts):
        parsed, warn, removed, raw_output = _extract_expressions_for_single_text(
            source_text,
            system_prompt=system_prompt,
            llama_cpp_url=llama_cpp_url,
            llama_cpp_model=llama_cpp_model,
            timeout_s=timeout_s,
            max_tokens=max_tokens,
            seed=seed,
        )
        if warn:
            warnings.append(f"{warn_prefix} {idx}: {warn}" if per_chunk_llm else f"{warn_prefix}: {warn}")
        filtered_out_total += removed
        extracted_by_source.append((parsed, warn, removed, raw_output))

    if per_chunk_llm:
        for idx, chunk_text in enumerate(chunks):
            parsed, _warn, _removed, raw_output = extracted_by_source[idx]
            by_chunk.append(
                {
                    "id": idx,
                    "chunk": chunk_text,
                    "expressions": parsed,
                    "request_fingerprint": _request_fingerprint(
                        text=chunk_text,
                        system_prompt=system_prompt,
                        llama_cpp_url=llama_cpp_url,
                        llama_cpp_model=llama_cpp_model,
                        timeout_s=timeout_s,
                        max_tokens=max_tokens,
                        seed=seed,
                    ),
                    **({"raw_llm_output": raw_output} if debug_return_raw else {}),
                }
            )
            all_expressions.extend(parsed)
    else:
        parsed_global, _warn, _removed, raw_global = extracted_by_source[0]
        all_expressions.extend(parsed_global)
        for idx, chunk_text in enumerate(chunks):
            chunk_exprs = [expr for expr in parsed_global if expr in chunk_text]
            by_chunk.append(
                {
                    "id": idx,
                    "chunk": chunk_text,
                    "expressions": chunk_exprs,
                    "request_fingerprint": _request_fingerprint(
                        text=chunk_text,
                        system_prompt=system_prompt,
                        llama_cpp_url=llama_cpp_url,
                        llama_cpp_model=llama_cpp_model,
                        timeout_s=timeout_s,
                        max_tokens=max_tokens,
                        seed=seed,
                    ),
                    **({"raw_llm_output": raw_global} if debug_return_raw else {}),
                }
            )

    dedup: list[str] = []
    seen: set[str] = set()
    for expr in all_expressions:
        if expr not in seen:
            dedup.append(expr)
            seen.add(expr)

    if filtered_out_total:
        warnings.append(f"filtered out {filtered_out_total} non-verbatim expressions")
    result = {
        "chunks": chunks,
        "by_chunk": by_chunk,
        "expressions": dedup,
        "request_fingerprint": _request_fingerprint(
            text=cleaned,
            system_prompt=system_prompt,
            llama_cpp_url=llama_cpp_url,
            llama_cpp_model=llama_cpp_model,
            timeout_s=timeout_s,
            max_tokens=max_tokens,
            seed=seed,
        ),
        "inference_params": {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 1,
            "repeat_penalty": 1.0,
            "seed": seed,
            "per_chunk_llm": per_chunk_llm,
        },
        "warning": " | ".join(warnings) if warnings else None,
    }
    if debug_return_raw:
        result["raw_llm_outputs"] = [entry.get("raw_llm_output", "") for entry in by_chunk]
    return result

def _extract_expressions_for_single_text(
    text: str,
    *,
    system_prompt: str,
    llama_cpp_url: str,
    llama_cpp_model: str,
    timeout_s: float,
    max_tokens: int,
    seed: int,
) -> tuple[list[str], str | None, int, str]:
    user_prompt = (
        "Quelles sont les expressions clés contenues à l'identique dans ce texte :\n"
        f"{text}"
    )
    payload = {
        "model": llama_cpp_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "repeat_penalty": 1.0,
        "seed": seed,
        "stream": False,
    }
    _append_extraction_prompt_log(
        llama_cpp_url=llama_cpp_url,
        llama_cpp_model=llama_cpp_model,
        max_tokens=max_tokens,
        timeout_s=timeout_s,
        seed=seed,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
    raw, req_warning = _post_chat_completion(llama_cpp_url=llama_cpp_url, payload=payload, timeout_s=timeout_s)
    if raw is None:
        return [], req_warning, 0, ""

    parsed, parse_warning = _parse_expressions_output(raw)
    if parsed is None:
        return [], parse_warning, 0, raw

    filtered, removed = _filter_verbatim_expressions(text, parsed)
    warning = _merge_warnings(parse_warning)
    return filtered, warning, removed, raw


def _load_prompt_text(prompt_path: str | None) -> str | None:
    candidates: list[Path] = []
    if prompt_path:
        p = Path(prompt_path)
        if p.is_absolute():
            candidates.append(p)
        else:
            candidates.append(Path.cwd() / p)
            candidates.append(SECRETARIUS_ROOT.parent / p)
            candidates.append(SECRETARIUS_ROOT / p)
    candidates.append(DEFAULT_PROMPT_PATH)
    candidates.append(SECRETARIUS_ROOT / "prompts" / "prompt.txt")

    for target in candidates:
        try:
            if target.exists():
                return target.read_text(encoding="utf-8").strip()
        except OSError:
            continue
    return None


def _load_chunker() -> tuple[Any | None, str | None]:
    global _CACHED_CHUNKER
    if _CACHED_CHUNKER is not None:
        return _CACHED_CHUNKER, None
    if not CHUNK_DATA_PATH.exists():
        return None, f"missing chunker module: {CHUNK_DATA_PATH}"

    try:
        nltk_data_path = resolve_nltk_data_path()
        if nltk_data_path is not None:
            os.environ.setdefault("NLTK_DATA", str(nltk_data_path))
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("SECRETARIUS_LOCAL_FILES_ONLY", "1")
        local_model_path = resolve_sentence_model_path()
        if local_model_path is not None:
            os.environ.setdefault("SECRETARIUS_SENTENCE_MODEL", str(local_model_path))

        spec = importlib.util.spec_from_file_location("secretarius_vendor_chunk_data", CHUNK_DATA_PATH)
        if spec is None or spec.loader is None:
            return None, "unable to load chunk_data.py spec"
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _CACHED_CHUNKER = module.SemanticChunker()
        return _CACHED_CHUNKER, None
    except Exception as exc:
        return None, f"unable to initialize SemanticChunker: {exc}"


def _post_chat_completion(*, llama_cpp_url: str, payload: dict[str, Any], timeout_s: float) -> tuple[str | None, str | None]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urlrequest.Request(
        llama_cpp_url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlrequest.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read()
    except urlerror.HTTPError as exc:
        return None, f"llama.cpp server request failed: HTTP Error {exc.code}: {exc.reason}"
    except (urlerror.URLError, TimeoutError) as exc:
        return None, f"llama.cpp server request failed: {exc}"

    try:
        parsed = json.loads(raw.decode("utf-8"))
        content = parsed["choices"][0]["message"]["content"]
    except Exception as exc:
        return None, f"invalid llama.cpp server response shape: {exc}"
    return str(content), None


def _parse_expressions_output(content: str) -> tuple[list[str] | None, str | None]:
    candidates = [content.strip()]
    match = _JSON_CODEBLOCK_RE.search(content or "")
    if match:
        candidates.append(match.group(1).strip())
    candidates.append(_extract_json(content))

    last_err = "no parseable json output"
    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except Exception as exc:
            last_err = str(exc)
            continue

        if isinstance(parsed, list):
            if all(isinstance(x, str) for x in parsed):
                return [x for x in parsed if x.strip()], None
            return None, "json output list must contain only strings"
        if isinstance(parsed, dict):
            # tolerate wrappers if model returns {"expressions":[...]}
            exprs = parsed.get("expressions")
            if isinstance(exprs, list) and all(isinstance(x, str) for x in exprs):
                return [x for x in exprs if x.strip()], "json wrapper object detected"
            return None, "json output object does not contain string list 'expressions'"
        if isinstance(parsed, str):
            # nested JSON string
            nested, nested_warn = _parse_expressions_output(parsed)
            if nested is not None:
                return nested, _merge_warnings("nested json string output", nested_warn)
            return None, nested_warn or "nested json string unparsable"

        return None, f"unsupported json type: {type(parsed).__name__}"

    # Fallback for truncated JSON (e.g. "Unterminated string ...").
    for candidate in candidates:
        recovered = _recover_string_list(candidate)
        if recovered:
            return recovered, _merge_warnings(
                f"unable to parse llm json output: {last_err}",
                "recovered from partial json string list",
            )
    return None, f"unable to parse llm json output: {last_err}"


def _extract_json(text: str) -> str:
    t = (text or "").strip()
    start = t.find("[")
    if start != -1:
        end = t.rfind("]")
        if end != -1 and end > start:
            return t[start : end + 1]
    start_obj = t.find("{")
    if start_obj != -1:
        end_obj = t.rfind("}")
        if end_obj != -1 and end_obj > start_obj:
            return t[start_obj : end_obj + 1]
    return t


def _recover_string_list(candidate: str) -> list[str]:
    t = (candidate or "").strip()
    if not t:
        return []
    # Prefer list segment to avoid picking object keys like "expressions".
    start = t.find("[")
    segment = t[start:] if start != -1 else t
    end = segment.rfind("]")
    if end != -1:
        segment = segment[: end + 1]

    out: list[str] = []
    seen: set[str] = set()
    for raw in _JSON_STRING_RE.findall(segment):
        try:
            value = json.loads(f'"{raw}"')
        except Exception:
            continue
        if not isinstance(value, str):
            continue
        value = value.strip()
        if not value or value in seen:
            continue
        out.append(value)
        seen.add(value)
    return out


def _filter_verbatim_expressions(text: str, expressions: list[str]) -> tuple[list[str], int]:
    out: list[str] = []
    removed = 0
    seen: set[str] = set()
    for expr in expressions:
        if not expr or expr in seen:
            continue
        if expr in text:
            out.append(expr)
            seen.add(expr)
        else:
            removed += 1
    return out, removed


def _merge_warnings(*warnings: str | None) -> str | None:
    parts = [w for w in warnings if isinstance(w, str) and w.strip()]
    if not parts:
        return None
    return " | ".join(parts)


def _request_fingerprint(
    *,
    text: str,
    system_prompt: str,
    llama_cpp_url: str,
    llama_cpp_model: str,
    timeout_s: float,
    max_tokens: int,
    seed: int,
) -> str:
    payload = {
        "text": text,
        "system_prompt": system_prompt,
        "llama_cpp_url": llama_cpp_url,
        "llama_cpp_model": llama_cpp_model,
        "timeout_s": timeout_s,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "repeat_penalty": 1.0,
        "seed": seed,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _append_extraction_prompt_log(
    *,
    llama_cpp_url: str,
    llama_cpp_model: str,
    max_tokens: int,
    timeout_s: float,
    seed: int,
    system_prompt: str,
    user_prompt: str,
) -> None:
    journal_path = os.environ.get("SECRETARIUS_JOURNAL_LOG", "logs/guichet.log").strip()
    if not journal_path:
        return

    channel = os.environ.get("SECRETARIUS_JOURNAL_CHANNEL", "mcp").strip() or "mcp"
    timestamp = datetime.now().isoformat(timespec="seconds")
    message = (
        "llama.cpp extract request\n"
        f"url={llama_cpp_url}\n"
        f"model={llama_cpp_model}\n"
        f"max_tokens={max_tokens}\n"
        f"timeout_s={timeout_s}\n"
        f"seed={seed}\n"
        f"system_prompt:\n{system_prompt}\n"
        f"user_prompt:\n{user_prompt}"
    )
    line = f"{timestamp}\t{channel}\tTHOUGHT\tASSISTANT\t{message}\n"

    try:
        path = Path(journal_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
    except OSError:
        LOGGER.debug("failed to append extraction prompt log", exc_info=True)
