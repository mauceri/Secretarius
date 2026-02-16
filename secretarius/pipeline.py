from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, List

import requests


@dataclass
class DocRecord:
    id_doc: str
    source: str
    titre: str
    contenu: str


@dataclass
class ChunkRecord:
    ordre_chunk: int
    chunk: str
    expressions_caracteristiques: List[str]


class SimpleSemanticChunker:
    """Deterministic chunker for phase-1 tests.

    This does not replace the production semantic chunker; it gives a stable
    contract for local tests and integration scaffolding.
    """

    SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

    def __init__(self, min_sentences: int = 2, max_sentences: int = 5) -> None:
        self.min_sentences = min_sentences
        self.max_sentences = max_sentences

    def split_sentences(self, text: str) -> List[str]:
        raw = " ".join((text or "").split())
        if not raw:
            return []
        return [s.strip() for s in self.SENT_SPLIT_RE.split(raw) if s.strip()]

    def chunk(self, text: str) -> List[str]:
        sentences = self.split_sentences(text)
        if not sentences:
            return []

        chunks: List[str] = []
        buf: List[str] = []
        for sent in sentences:
            buf.append(sent)
            if len(buf) >= self.max_sentences:
                chunks.append(" ".join(buf))
                buf = []

        if buf:
            if chunks and len(buf) < self.min_sentences:
                chunks[-1] = f"{chunks[-1]} {' '.join(buf)}".strip()
            else:
                chunks.append(" ".join(buf))

        return [c for c in chunks if c.strip()]


class LlamaExtractorClient:
    def __init__(self, base_url: str = "http://localhost:8080", timeout_s: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def extract(self, chunk: str) -> List[str]:
        prompt = (
            "Extrais les expressions caractéristiques du texte ci-dessous. "
            "Réponds uniquement en JSON sous forme de tableau de chaînes.\n\n"
            f"Texte: {chunk}"
        )
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
        }
        r = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            timeout=self.timeout_s,
        )
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        return _parse_expressions(content)


_JSON_ARRAY_RE = re.compile(r"\[[\s\S]*\]")


def _parse_expressions(content: str) -> List[str]:
    content = (content or "").strip()
    match = _JSON_ARRAY_RE.search(content)
    text = match.group(0) if match else content
    parsed = json.loads(text)
    if not isinstance(parsed, list):
        raise ValueError("Extractor output is not a JSON array")
    return [str(x).strip() for x in parsed if str(x).strip()]


def index_document(doc: DocRecord, chunker: SimpleSemanticChunker, extractor: Any) -> List[ChunkRecord]:
    chunks = chunker.chunk(doc.contenu)
    out: List[ChunkRecord] = []
    for i, chunk in enumerate(chunks):
        exprs = extractor.extract(chunk)
        out.append(
            ChunkRecord(
                ordre_chunk=i,
                chunk=chunk,
                expressions_caracteristiques=exprs,
            )
        )
    return out


def prepare_query(query_text: str, chunker: SimpleSemanticChunker, extractor: Any) -> List[ChunkRecord]:
    pseudo_doc = DocRecord(id_doc="query", source="query", titre="query", contenu=query_text)
    return index_document(pseudo_doc, chunker, extractor)


def records_to_contract_json(records: List[ChunkRecord]) -> List[dict]:
    """Return the phase-1 JSON contract for chunk extraction."""
    payload: List[dict] = []
    for expected_order, record in enumerate(records):
        _validate_chunk_record(record, expected_order)
        payload.append(
            {
                "ordre_chunk": record.ordre_chunk,
                "chunk": record.chunk,
                "expressions_caracteristiques": list(record.expressions_caracteristiques),
            }
        )
    return payload


def _validate_chunk_record(record: ChunkRecord, expected_order: int) -> None:
    if not isinstance(record, ChunkRecord):
        raise TypeError("Each item must be a ChunkRecord")
    if not isinstance(record.ordre_chunk, int):
        raise TypeError("ordre_chunk must be an integer")
    if record.ordre_chunk != expected_order:
        raise ValueError("ordre_chunk must match list order (0..n-1)")
    if not isinstance(record.chunk, str):
        raise TypeError("chunk must be a string")
    if not record.chunk.strip():
        raise ValueError("chunk must be a non-empty string")
    if not isinstance(record.expressions_caracteristiques, list):
        raise TypeError("expressions_caracteristiques must be a list")
    if any(not isinstance(x, str) for x in record.expressions_caracteristiques):
        raise TypeError("expressions_caracteristiques items must be strings")
