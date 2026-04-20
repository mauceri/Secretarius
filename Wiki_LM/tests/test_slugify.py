from ingest import _slugify


def test_ascii():
    assert _slugify("Gerard Salton") == "gerard-salton"


def test_accents():
    assert _slugify("Théorie de l'information") == "theorie-de-l-information"


def test_special_chars():
    assert _slugify("BM25 (Okapi)") == "bm25-okapi"


def test_truncation():
    long = "a" * 100
    assert len(_slugify(long)) <= 60


def test_empty():
    assert _slugify("") == ""


def test_only_special():
    assert _slugify("---") == ""


def test_numbers():
    assert _slugify("TF-IDF v2") == "tf-idf-v2"
