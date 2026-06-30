from pathlib import Path
from unittest.mock import MagicMock
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_generate_one_structure():
    from generate_corpus import generate_one
    mock_result = MagicMock()
    mock_result.text = "garde cet article https://example.com"
    mock_result.command = "/c"
    mock_result.args = "https://example.com"
    mock_predict = MagicMock(return_value=mock_result)

    entry = generate_one(mock_predict, "wiki_capture", "familier", "url_seule")

    assert entry["text"] == "garde cet article https://example.com"
    assert entry["intention"] == "wiki_capture"
    assert entry["registre"] == "familier"
    assert entry["variante"] == "url_seule"
    assert entry["action"]["command"] == "/c"
    assert entry["action"]["args"] == "https://example.com"


def test_generate_one_null_command():
    from generate_corpus import generate_one
    mock_result = MagicMock()
    mock_result.text = "commande une pizza"
    mock_result.command = "null"
    mock_result.args = ""
    mock_predict = MagicMock(return_value=mock_result)

    entry = generate_one(mock_predict, "out_of_scope", "familier", "action_impossible")
    assert entry["action"]["command"] is None
