from http import HTTPStatus

from secretarius.server import process_extract_payload


class StubExtractor:
    def extract(self, chunk: str):  # noqa: ANN201
        return [chunk.split()[0].lower()] if chunk.split() else []


def test_process_extract_payload_success_contract_shape() -> None:
    status, body = process_extract_payload(
        {
            "text": "Le camail est vert. Le voile est blanc.",
            "min_sentences": 1,
            "max_sentences": 1,
        },
        extractor=StubExtractor(),
    )

    assert status == HTTPStatus.OK
    assert "results" in body
    assert isinstance(body["results"], list)
    assert body["results"]
    assert set(body["results"][0].keys()) == {
        "ordre_chunk",
        "chunk",
        "expressions_caracteristiques",
    }


def test_process_extract_payload_rejects_empty_text() -> None:
    status, body = process_extract_payload({"text": ""}, extractor=StubExtractor())
    assert status == HTTPStatus.BAD_REQUEST
    assert "error" in body


def test_process_extract_payload_rejects_invalid_chunk_bounds() -> None:
    status, body = process_extract_payload(
        {"text": "Texte valide.", "min_sentences": 3, "max_sentences": 1},
        extractor=StubExtractor(),
    )
    assert status == HTTPStatus.BAD_REQUEST
    assert "error" in body
