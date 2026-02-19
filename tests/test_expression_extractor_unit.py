from secretarius.expression_extractor import _filter_verbatim_expressions


def test_filter_verbatim_expressions_keeps_only_exact_substrings() -> None:
    chunk = "Le camail vert contraste avec le voile blanc."
    expressions = [
        "camail vert",
        "voile blanc",
        "camail rouge",
        "Contraste",
        "",
        "voile blanc",
    ]
    kept, removed = _filter_verbatim_expressions(chunk, expressions)
    assert kept == ["camail vert", "voile blanc"]
    assert removed == 3

