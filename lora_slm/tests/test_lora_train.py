import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import subprocess
from datasets import DatasetDict, Dataset


def _ds(messages_list):
    return DatasetDict({"train": Dataset.from_list(
        [{"messages": m} for m in messages_list]
    )})


def test_build_text_dataset_chatML():
    from lora_train import build_text_dataset
    ds = _ds([[
        {"role": "system",    "content": "Routeur Tiron."},
        {"role": "user",      "content": "capture https://example.com"},
        {"role": "assistant", "content": '{"command": "/c", "args": "https://example.com"}'},
    ]])
    result = build_text_dataset(ds, "", "", num_proc=1)
    text = result["train"][0]["text"]
    assert "<|system|>: Routeur Tiron." in text
    assert "<|user|>: capture https://example.com" in text
    assert '<|assistant|>: {"command": "/c"' in text


def test_build_text_dataset_null_command():
    from lora_train import build_text_dataset
    ds = _ds([[
        {"role": "system",    "content": "Routeur Tiron."},
        {"role": "user",      "content": "commande une pizza"},
        {"role": "assistant", "content": '{"command": null, "args": ""}'},
    ]])
    result = build_text_dataset(ds, "", "", num_proc=1)
    text = result["train"][0]["text"]
    assert '"command": null' in text


def test_build_text_dataset_rejects_unknown_format():
    import pytest
    from lora_train import build_text_dataset
    ds = DatasetDict({"train": Dataset.from_list([{"unknown": "x"}])})
    with pytest.raises(ValueError):
        build_text_dataset(ds, "", "", num_proc=1)


def test_merge_and_quantize_importable():
    from merge_and_quantize import merge_lora, quantize_gguf
    assert callable(merge_lora)
    assert callable(quantize_gguf)


def test_merge_and_quantize_cli_help():
    result = subprocess.run(
        [sys.executable, "merge_and_quantize.py", "--help"],
        capture_output=True, text=True,
        cwd=str(Path(__file__).parent.parent)
    )
    assert "--base" in result.stdout
    assert "--lora" in result.stdout
    assert "--gguf-dir" in result.stdout
