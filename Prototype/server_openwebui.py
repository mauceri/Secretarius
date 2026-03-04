#!/usr/bin/env python3

import asyncio
import logging
from pathlib import Path
import socket

import uvicorn

from adapters.input.guichet_unique import GuichetUnique
from app_runtime import build_runtime, load_config
from openwebui_api import create_openwebui_app


def _can_bind(host: str, port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
            return True
    except OSError:
        return False


def _pick_openwebui_port(host: str, preferred_port: int, max_attempts: int) -> int:
    for offset in range(max_attempts):
        candidate = preferred_port + offset
        if _can_bind(host, candidate):
            if candidate != preferred_port:
                logging.getLogger(__name__).warning(
                    "openwebui port %d unavailable, using fallback port %d",
                    preferred_port,
                    candidate,
                )
            return candidate
    raise RuntimeError(
        f"No available OpenWebUI port in range {preferred_port}-{preferred_port + max_attempts - 1}"
    )


async def _main() -> None:
    project_root = Path(__file__).parent
    config_path = project_root / "config.yaml"
    config = load_config(str(config_path))

    webui_cfg = config.get("openwebui", {})
    journal_file = webui_cfg.get("journal_file") or "logs/openwebui.log"
    journal_path = str((project_root / journal_file).resolve())
    model_id = webui_cfg.get("model_id") or "secretarius-agent"
    host = webui_cfg.get("host") or "0.0.0.0"
    preferred_port = int(webui_cfg.get("port") or 8000)
    port_fallback_attempts = int(webui_cfg.get("port_fallback_attempts") or 3)
    port = _pick_openwebui_port(host, preferred_port, port_fallback_attempts)
    request_timeout_s = float(webui_cfg.get("request_timeout_s") or 90.0)

    guichet = GuichetUnique(journal_path=journal_path)
    runtime = await build_runtime(guichet, config_path=config_path)
    app = create_openwebui_app(
        gateway=guichet,
        model_id=model_id,
        request_timeout_s=request_timeout_s,
    )

    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, reload=False))
    try:
        await server.serve()
    finally:
        await runtime["mcp_client"].disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    asyncio.run(_main())
