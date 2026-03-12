#!/usr/bin/env python3

import asyncio
import logging
from pathlib import Path

import uvicorn

from adapters.input.guichet_unique import GuichetUnique
from app_runtime import build_runtime, load_config
from session_messenger_api import create_session_messenger_app


async def _main() -> None:
    project_root = Path(__file__).parent
    config_path = project_root / "config.yaml"
    config = load_config(str(config_path))

    session_cfg = config.get("session_messenger", {})
    journal_file = session_cfg.get("journal_file") or "logs/session_messenger.log"
    journal_path = str((project_root / journal_file).resolve())
    host = session_cfg.get("host") or "127.0.0.1"
    port = int(session_cfg.get("port") or 8002)
    request_timeout_s = float(session_cfg.get("request_timeout_s") or 120.0)

    guichet = GuichetUnique(
        journal_path=journal_path,
        channel_journal_paths={"session_messenger": journal_path},
    )
    runtime = await build_runtime(guichet, config_path=config_path)
    app = create_session_messenger_app(
        gateway=guichet,
        request_timeout_s=request_timeout_s,
    )
    logging.getLogger(__name__).info(
        "Starting Session Messenger API on http://%s:%d/session/message",
        host,
        port,
    )

    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, reload=False))
    try:
        await server.serve()
    finally:
        await runtime["mcp_client"].disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    asyncio.run(_main())
