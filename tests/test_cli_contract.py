import json

from secretarius import cli


def test_cli_outputs_contract_json(monkeypatch, capsys) -> None:
    class StubExtractor:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def extract(self, chunk: str):  # noqa: ANN201
            return [w.lower() for w in chunk.split()[:2]]

    monkeypatch.setattr(cli, "LlamaExtractorClient", StubExtractor)

    exit_code = cli.main(
        [
            "--text",
            "Le camail est vert. Le voile est blanc.",
            "--min-sentences",
            "1",
            "--max-sentences",
            "1",
        ]
    )
    assert exit_code == 0

    out = capsys.readouterr().out.strip()
    payload = json.loads(out)
    assert isinstance(payload, list)
    assert payload
    assert set(payload[0].keys()) == {"ordre_chunk", "chunk", "expressions_caracteristiques"}
    assert payload[0]["ordre_chunk"] == 0
    assert isinstance(payload[0]["chunk"], str)
    assert isinstance(payload[0]["expressions_caracteristiques"], list)
