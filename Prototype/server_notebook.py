#!/usr/bin/env python3

import asyncio
import logging
from pathlib import Path

import uvicorn

from adapters.input.guichet_unique import GuichetUnique
from app_runtime import build_runtime, load_config
from notebook_api import create_notebook_app


async def _main() -> None:
    project_root = Path(__file__).parent
    config_path = project_root / "config.yaml"
    config = load_config(str(config_path))

    notebook_cfg = config.get("notebook_api", {})
    journal_file = notebook_cfg.get("journal_file") or "logs/notebook.log"
    journal_path = str((project_root / journal_file).resolve())
    model_id = notebook_cfg.get("model_id") or "secretarius-notebook"
    host = notebook_cfg.get("host") or "0.0.0.0"
    port = int(notebook_cfg.get("port") or 8001)
    request_timeout_s = float(notebook_cfg.get("request_timeout_s") or 120.0)

    guichet = GuichetUnique(journal_path=journal_path)
    runtime = await build_runtime(guichet, config_path=config_path)
    app = create_notebook_app(
        gateway=guichet,
        model_id=model_id,
        request_timeout_s=request_timeout_s,
    )
    logging.getLogger(__name__).info("Starting Notebook API on http://%s:%d/v1", host, port)

    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, reload=False))
    try:
        await server.serve()
    finally:
        await runtime["mcp_client"].disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    asyncio.run(_main())
