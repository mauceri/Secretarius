import argparse
import json
import time

import requests


def bench(base_url: str, prompt_path: str, n: int, api_key: str | None):
    body = json.load(open(prompt_path))
    body["stream"] = True
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    rows = []
    for i in range(n):
        t0 = time.perf_counter()
        ttft = None
        chunks = 0
        with requests.post(
            f"{base_url}/v1/chat/completions",
            json=body, headers=headers, stream=True, timeout=600,
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line or not line.startswith(b"data: "):
                    continue
                data = line[6:]
                if data == b"[DONE]":
                    break
                if ttft is None:
                    ttft = time.perf_counter() - t0
                try:
                    if json.loads(data)["choices"][0]["delta"].get("content"):
                        chunks += 1  # approx : 1 chunk SSE ~ 1 token
                except Exception:
                    pass
        total = time.perf_counter() - t0
        gen = max(total - (ttft or 0), 1e-6)
        toks = chunks / gen
        rows.append((i, ttft or 0.0, total, chunks, toks))
        tag = "FROID" if i == 0 else "chaud"
        print(f"[{tag}] req{i}: TTFT={ttft:.2f}s total={total:.2f}s out~{chunks} {toks:.1f} tok/s")

    warm = rows[1:] or rows
    m = lambda k: sum(x[k] for x in warm) / len(warm)
    print(f"\nRESUME {base_url}")
    print(f"  froid : TTFT={rows[0][1]:.2f}s total={rows[0][2]:.2f}s")
    print(f"  chaud : TTFT~{m(1):.2f}s total~{m(2):.2f}s {m(4):.1f} tok/s (n={len(warm)})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("base_url")
    p.add_argument("--prompt", default="tiron_modal/prompt.json")
    p.add_argument("-n", type=int, default=6)
    p.add_argument("--api-key", default=None)
    a = p.parse_args()
    bench(a.base_url, a.prompt, a.n, a.api_key)
