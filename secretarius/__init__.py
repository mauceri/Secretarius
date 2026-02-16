"""Secretarius phase-1 scaffolding."""

__all__ = [
    "ChunkRecord",
    "DocRecord",
    "SimpleSemanticChunker",
    "LlamaExtractorClient",
    "index_document",
    "prepare_query",
    "records_to_contract_json",
    "process_extract_payload",
    "run_server",
]

from .pipeline import (
    ChunkRecord,
    DocRecord,
    LlamaExtractorClient,
    SimpleSemanticChunker,
    index_document,
    prepare_query,
    records_to_contract_json,
)
from .server import process_extract_payload, run_server
