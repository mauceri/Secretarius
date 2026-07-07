import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from to_lora_format_qa import convert_entry_qa, SYSTEM_PROMPT_QA


def test_convert_entry_qa_chatml():
    entry = {"document_id": "config", "document": "DOC", "question": "Q ?",
             "answer": "R.", "type_question": "factuelle", "registre": "poli"}
    out = convert_entry_qa(entry)
    msgs = out["messages"]
    assert msgs[0] == {"role": "system", "content": SYSTEM_PROMPT_QA}
    assert msgs[1] == {"role": "user", "content": "Document:\nDOC\n\nQuestion: Q ?"}
    assert msgs[2] == {"role": "assistant", "content": "R."}
