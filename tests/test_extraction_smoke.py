import os

import pytest

from secretarius.pipeline import LlamaExtractorClient


@pytest.mark.smoke
@pytest.mark.integration
def test_llama_extractor_smoke() -> None:
    if os.getenv("RUN_LLAMA_SMOKE") != "1":
        pytest.skip("Set RUN_LLAMA_SMOKE=1 to run llama-server smoke test")

    base_url = os.getenv("LLAMA_BASE_URL", "http://localhost:8080")
    client = LlamaExtractorClient(base_url=base_url, timeout_s=60)
    text = "Le camail est de soie verte et le voile blanc couvre le front."
    expressions = client.extract(text)

    assert isinstance(expressions, list)
    assert all(isinstance(x, str) and x.strip() for x in expressions)
