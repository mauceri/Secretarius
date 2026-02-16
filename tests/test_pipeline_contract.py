import pytest

from secretarius.pipeline import (
    ChunkRecord,
    DocRecord,
    SimpleSemanticChunker,
    index_document,
    prepare_query,
    records_to_contract_json,
)


def test_index_document_contract_fields(fake_extractor) -> None:
    doc = DocRecord(
        id_doc="doc-1",
        source="tests",
        titre="Document test",
        contenu="Le camail est vert. Le voile est blanc. Les prunelles brillent.",
    )
    chunker = SimpleSemanticChunker(min_sentences=1, max_sentences=2)
    records = index_document(doc, chunker, fake_extractor)

    assert records
    first = records[0]
    assert isinstance(first.ordre_chunk, int)
    assert isinstance(first.chunk, str)
    assert isinstance(first.expressions_caracteristiques, list)
    assert first.ordre_chunk == 0


def test_prepare_query_uses_same_contract(fake_extractor) -> None:
    chunker = SimpleSemanticChunker(min_sentences=1, max_sentences=2)
    records = prepare_query("Trouver les passages sur le voile blanc.", chunker, fake_extractor)
    assert records
    assert records[0].ordre_chunk == 0
    assert isinstance(records[0].expressions_caracteristiques, list)


def test_records_to_contract_json_shape(fake_extractor) -> None:
    doc = DocRecord(
        id_doc="doc-1",
        source="tests",
        titre="Document test",
        contenu="Le camail est vert. Le voile est blanc.",
    )
    chunker = SimpleSemanticChunker(min_sentences=1, max_sentences=1)
    records = index_document(doc, chunker, fake_extractor)
    payload = records_to_contract_json(records)

    assert isinstance(payload, list)
    assert payload
    assert set(payload[0].keys()) == {"ordre_chunk", "chunk", "expressions_caracteristiques"}
    assert payload[0]["ordre_chunk"] == 0
    assert isinstance(payload[0]["chunk"], str)
    assert isinstance(payload[0]["expressions_caracteristiques"], list)


def test_records_to_contract_json_rejects_non_sequential_order() -> None:
    records = [
        ChunkRecord(
            ordre_chunk=1,
            chunk="Texte valide.",
            expressions_caracteristiques=["texte"],
        )
    ]
    with pytest.raises(ValueError, match="ordre_chunk"):
        records_to_contract_json(records)


def test_records_to_contract_json_rejects_non_string_expression() -> None:
    records = [
        ChunkRecord(
            ordre_chunk=0,
            chunk="Texte valide.",
            expressions_caracteristiques=["ok", 12],  # type: ignore[list-item]
        )
    ]
    with pytest.raises(TypeError, match="items must be strings"):
        records_to_contract_json(records)
