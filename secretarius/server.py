from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Tuple

from .pipeline import (
    DocRecord,
    LlamaExtractorClient,
    SimpleSemanticChunker,
    index_document,
    records_to_contract_json,
)


def process_extract_payload(
    payload: Dict[str, Any],
    extractor: Any,
) -> Tuple[int, Dict[str, Any]]:
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        return HTTPStatus.BAD_REQUEST, {"error": "Field 'text' must be a non-empty string"}

    min_sentences = payload.get("min_sentences", 2)
    max_sentences = payload.get("max_sentences", 5)
    if not isinstance(min_sentences, int) or not isinstance(max_sentences, int):
        return HTTPStatus.BAD_REQUEST, {"error": "Fields 'min_sentences' and 'max_sentences' must be integers"}
    if min_sentences < 1 or max_sentences < 1:
        return HTTPStatus.BAD_REQUEST, {"error": "Fields 'min_sentences' and 'max_sentences' must be >= 1"}
    if min_sentences > max_sentences:
        return HTTPStatus.BAD_REQUEST, {"error": "'min_sentences' cannot be greater than 'max_sentences'"}

    chunker = SimpleSemanticChunker(min_sentences=min_sentences, max_sentences=max_sentences)
    doc = DocRecord(id_doc="http-input", source="http", titre="http", contenu=text)
    records = index_document(doc, chunker, extractor)
    return HTTPStatus.OK, {"results": records_to_contract_json(records)}


class SecretariusHandler(BaseHTTPRequestHandler):
    extractor: Any = LlamaExtractorClient()

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/extract":
            self._write_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(content_length)
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("JSON body must be an object")
        except (ValueError, json.JSONDecodeError):
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid JSON payload"})
            return

        status, response = process_extract_payload(payload, self.extractor)
        self._write_json(status, response)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._write_json(HTTPStatus.OK, {"status": "ok"})
            return
        self._write_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _write_json(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run_server(host: str = "0.0.0.0", port: int = 8090) -> None:
    server = ThreadingHTTPServer((host, port), SecretariusHandler)
    server.serve_forever()

