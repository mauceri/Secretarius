from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DEFAULT_LOCALE = "fr"
LOCALES_DIR = Path(__file__).resolve().parent / "locales"


def _load_locale_payload(locale: str) -> dict[str, Any]:
    normalized = (locale or DEFAULT_LOCALE).strip().lower() or DEFAULT_LOCALE
    path = LOCALES_DIR / f"{normalized}.yaml"
    if not path.exists():
        path = LOCALES_DIR / f"{DEFAULT_LOCALE}.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


class Translator:
    def __init__(self, locale: str = DEFAULT_LOCALE):
        self.locale = (locale or DEFAULT_LOCALE).strip().lower() or DEFAULT_LOCALE
        self._payload = _load_locale_payload(self.locale)
        self._fallback = _load_locale_payload(DEFAULT_LOCALE) if self.locale != DEFAULT_LOCALE else self._payload

    def get(self, key: str, **kwargs: Any) -> str:
        template = _lookup_key(self._payload, key)
        if template is None:
            template = _lookup_key(self._fallback, key)
        if not isinstance(template, str):
            return key
        if kwargs:
            return template.format(**kwargs)
        return template


def _lookup_key(payload: dict[str, Any], key: str) -> Any:
    current: Any = payload
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current
