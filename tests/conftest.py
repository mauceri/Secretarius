from typing import List

import pytest


class FakeExtractor:
    def extract(self, chunk: str) -> List[str]:
        words = [w.strip(" ,.;:!?()[]\"'").lower() for w in chunk.split()]
        words = [w for w in words if len(w) >= 5]
        uniq = []
        seen = set()
        for w in words:
            if w not in seen:
                seen.add(w)
                uniq.append(w)
        return uniq[:8]


@pytest.fixture
def fake_extractor() -> FakeExtractor:
    return FakeExtractor()
