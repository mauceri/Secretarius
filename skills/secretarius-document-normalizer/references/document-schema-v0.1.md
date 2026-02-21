# secretarius.document.v0.1 (résumé de travail)

```json
{
  "schema": "secretarius.document.v0.1",
  "doc_id": "string|null",
  "type": "url|note|article|pdf|email|chat|file|snippet|bookmark|other",
  "source": {
    "url": "string|null",
    "title": "string|null",
    "authors": ["string"],
    "publisher": "string|null",
    "language": "string|null",
    "retrieved_at": "ISO-8601|null",
    "canonical_url": "string|null",
    "source_id": "string|null"
  },
  "content": {
    "mode": "none|inline|ref",
    "text": "string|null",
    "content_ref": "string|null",
    "content_mime": "string|null",
    "hash": "string|null",
    "length_chars": "int|null"
  },
  "user_fields": {
    "theme": "string|null",
    "keywords": ["string"],
    "tags": ["string"],
    "status": "draft|pending|verified|archived|temporary|deleted",
    "expires_at": "ISO-8601|null",
    "created_at": "ISO-8601|null",
    "updated_at": "ISO-8601|null",
    "confidence_user": "low|medium|high|null",
    "notes": "string|null"
  },
  "derived": {
    "theme_inferred": { "value": "string", "confidence": "0..1" },
    "summary": "string|null",
    "chunks": [
      { "chunk_id": "string", "start": "int", "end": "int", "text_ref": "string|null" }
    ],
    "expressions": [
      {
        "expression": "string",
        "span": [0, 0],
        "weight": "0..1|null",
        "norm": "string|null",
        "embedding_ref": "string|null"
      }
    ]
  },
  "indexing": {
    "pipeline_version": "string|null",
    "state": "new|queued|fetching|extracting|embedding|upserting|done|error",
    "job_id": "string|null",
    "errors": [
      { "at": "ISO-8601", "stage": "string", "message": "string" }
    ]
  }
}
```

## Contrainte minimale d'entrée

- `type` obligatoire
- et au moins un de:
  - `source.url`
  - `content.text`
  - `content.content_ref`

