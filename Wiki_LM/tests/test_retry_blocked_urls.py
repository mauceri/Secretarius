from __future__ import annotations

from pathlib import Path

from retry_blocked_urls import retry_all


def test_retries_error_files_and_marks_success(ingestor, raw_dir: Path, monkeypatch):
    (raw_dir / "20260101-000000-example-com.url.error").write_text(
        "https://example.com/article\n", encoding="utf-8"
    )

    def fake_ingest(self, source, **kwargs):
        return "src-article"

    monkeypatch.setattr(type(ingestor), "ingest", fake_ingest)

    results = retry_all(ingestor, raw_dir, dry_run=False)

    assert results[0]["status"] == "ingested"
    assert not (raw_dir / "20260101-000000-example-com.url.error").exists()


def test_retries_error_files_and_keeps_failure_with_reason(ingestor, raw_dir: Path, monkeypatch):
    (raw_dir / "20260101-000000-example-com.url.error").write_text(
        "https://example.com/article\n", encoding="utf-8"
    )

    def failing_ingest(self, source, **kwargs):
        raise ConnectionError("DNS lookup failed")

    monkeypatch.setattr(type(ingestor), "ingest", failing_ingest)

    results = retry_all(ingestor, raw_dir, dry_run=False)

    assert results[0]["status"] == "failed"
    assert "DNS lookup failed" in results[0]["error"]
    content = (raw_dir / "20260101-000000-example-com.url.error").read_text(encoding="utf-8")
    assert "DNS lookup failed" in content


def test_dry_run_does_not_call_ingest(ingestor, raw_dir: Path, monkeypatch):
    (raw_dir / "20260101-000000-example-com.url.error").write_text(
        "https://example.com/article\n", encoding="utf-8"
    )
    called = []

    def tracking_ingest(self, source, **kwargs):
        called.append(source)
        return "src-article"

    monkeypatch.setattr(type(ingestor), "ingest", tracking_ingest)

    retry_all(ingestor, raw_dir, dry_run=True)

    assert called == []
