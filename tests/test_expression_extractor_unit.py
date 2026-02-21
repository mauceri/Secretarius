from secretarius.expression_extractor import _filter_verbatim_expressions, _parse_expressions_output


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
    assert removed == 2


def test_parse_expressions_output_recovers_from_truncated_json_list() -> None:
    raw = '["têtes entassées", "charnier", "chambre aux deniers", "porte-paniers", "é'
    parsed, warning = _parse_expressions_output(raw)
    assert parsed == ["têtes entassées", "charnier", "chambre aux deniers", "porte-paniers"]
    assert warning is not None
    assert "recovered from partial json string list" in warning
