#!/usr/bin/env python3
"""
scout_process.py — Traite un fichier tâche scout après fetch curl.
Appelle injection-guard (/check), injecte le résultat dans le fichier tâche.

Usage (URL)   : python3 scout_process.py <task_file> <fetched_html_file>
Usage (email) : python3 scout_process.py <task_file>
Exit : toujours 0 (fail-safe — les erreurs sont encodées dans fetched_content)
"""
import json
import sys

import requests

GUARD_URL = "http://localhost:8990/check"
GUARD_TIMEOUT = 3
MAX_RAW_LEN = 200_000


def process(task_file: str, fetched_file: str = None) -> int:
    with open(task_file) as f:
        data = json.load(f)

    if fetched_file is not None:
        with open(fetched_file, errors='replace') as f:
            raw = f.read()
        content_type = "html"
        content = raw[:MAX_RAW_LEN]
    elif 'check_email' in data:
        content_type = "text"
        content = data['check_email'][:MAX_RAW_LEN]
    else:
        content_type = "text"
        content = ""

    try:
        resp = requests.post(
            GUARD_URL,
            json={"type": content_type, "content": content},
            timeout=GUARD_TIMEOUT,
        )
        resp.raise_for_status()
        result = resp.json()
    except Exception:
        result = {"blocked": True, "reason": "injection-guard unavailable"}

    data['fetched_content'] = json.dumps(result, ensure_ascii=False)
    with open(task_file, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    if not result.get('blocked'):
        print(len(result.get('clean_text', '')))
    return 0


if __name__ == '__main__':
    fetched = sys.argv[2] if len(sys.argv) > 2 else None
    sys.exit(process(sys.argv[1], fetched))
