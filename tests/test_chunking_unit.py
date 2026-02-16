from secretarius.pipeline import SimpleSemanticChunker


def test_chunking_on_empty_text_returns_empty_list() -> None:
    chunker = SimpleSemanticChunker()
    assert chunker.chunk("") == []


def test_chunking_splits_long_multisentence_text() -> None:
    chunker = SimpleSemanticChunker(min_sentences=2, max_sentences=3)
    text = (
        "Le chat observe la ville. "
        "La pluie tombe sur les toits. "
        "Le vent tourne au nord. "
        "Un passant ferme son manteau. "
        "La nuit progresse rapidement. "
        "Les lampes s'allument sur le quai."
    )
    chunks = chunker.chunk(text)
    assert len(chunks) >= 2
    assert all(chunk.strip() for chunk in chunks)
