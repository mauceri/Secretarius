#!/usr/bin/env python3

import asyncio
import logging
from pathlib import Path
from typing import Any

import uvicorn

from adapters.input.guichet_unique import GuichetUnique
from app_runtime import build_runtime, load_config
from notebook_api import create_notebook_app
from openwebui_api import create_openwebui_app
from session_messenger_api import create_session_messenger_app
from tui_api import create_tui_app

logger = logging.getLogger(__name__)


async def _run_component(
    name: str,
    coro,
    stop_event: asyncio.Event,
    component_errors: list[tuple[str, BaseException]],
) -> None:
    try:
        await coro
        logger.info("%s stopped", name)
    except asyncio.CancelledError:
        raise
    except BaseException as exc:
        component_errors.append((name, exc))
        logger.exception("%s failed", name)
    finally:
        stop_event.set()


async def main() -> None:
    project_root = Path(__file__).parent
    config_path = project_root / "config.yaml"
    config = load_config(str(config_path))

    ui_cfg = config.get("ui", {})
    webui_cfg = config.get("openwebui", {})
    notebook_cfg = config.get("notebook_api", {})
    session_cfg = config.get("session_messenger", {})
    tui_cfg = config.get("tui_api", {})

    journal_file = ui_cfg.get("journal_file") or "logs/guichet.log"
    openwebui_journal_file = webui_cfg.get("journal_file") or "logs/openwebui.log"
    notebook_journal_file = notebook_cfg.get("journal_file") or "logs/notebook.log"
    session_journal_file = session_cfg.get("journal_file") or "logs/session_messenger.log"
    tui_journal_file = tui_cfg.get("journal_file") or journal_file

    journal_path = str((project_root / journal_file).resolve())
    openwebui_journal_path = str((project_root / openwebui_journal_file).resolve())
    notebook_journal_path = str((project_root / notebook_journal_file).resolve())
    session_journal_path = str((project_root / session_journal_file).resolve())
    tui_journal_path = str((project_root / tui_journal_file).resolve())

    host = webui_cfg.get("host") or "0.0.0.0"
    port = int(webui_cfg.get("port") or 8000)
    model_id = webui_cfg.get("model_id") or "secretarius-agent"
    request_timeout_s = float(webui_cfg.get("request_timeout_s") or 90.0)
    notebook_enabled = bool(notebook_cfg.get("enabled", True))
    notebook_host = notebook_cfg.get("host") or "0.0.0.0"
    notebook_port = int(notebook_cfg.get("port") or 8001)
    notebook_model_id = notebook_cfg.get("model_id") or "secretarius-notebook"
    notebook_request_timeout_s = float(notebook_cfg.get("request_timeout_s") or 120.0)
    session_enabled = bool(session_cfg.get("enabled", True))
    session_host = session_cfg.get("host") or "127.0.0.1"
    session_port = int(session_cfg.get("port") or 8002)
    session_request_timeout_s = float(session_cfg.get("request_timeout_s") or 120.0)
    tui_enabled = bool(tui_cfg.get("enabled", True))
    tui_host = tui_cfg.get("host") or "127.0.0.1"
    tui_port = int(tui_cfg.get("port") or 8003)
    tui_request_timeout_s = float(tui_cfg.get("request_timeout_s") or 120.0)

    guichet = GuichetUnique(
        journal_path=journal_path,
        channel_journal_paths={
            "tui": tui_journal_path,
            "openwebui": openwebui_journal_path,
            "notebook": notebook_journal_path,
            "session_messenger": session_journal_path,
        },
    )

    runtime: dict[str, Any] | None = None
    openwebui_server: uvicorn.Server | None = None
    notebook_server: uvicorn.Server | None = None
    session_server: uvicorn.Server | None = None
    tui_server: uvicorn.Server | None = None
    tasks: list[asyncio.Task] = []
    stop_event = asyncio.Event()
    component_errors: list[tuple[str, BaseException]] = []

    try:
        runtime = await build_runtime(guichet, config_path=config_path)

        openwebui_app = create_openwebui_app(
            gateway=guichet,
            model_id=model_id,
            request_timeout_s=request_timeout_s,
        )
        logger.info("Starting OpenAI-compatible API on http://%s:%d/v1", host, port)
        openwebui_server = uvicorn.Server(uvicorn.Config(openwebui_app, host=host, port=port, reload=False))
        tasks.append(
            asyncio.create_task(
                _run_component("openwebui_api", openwebui_server.serve(), stop_event, component_errors)
            )
        )

        if notebook_enabled:
            notebook_app = create_notebook_app(
                gateway=guichet,
                model_id=notebook_model_id,
                request_timeout_s=notebook_request_timeout_s,
            )
            logger.info("Starting Notebook API on http://%s:%d/v1", notebook_host, notebook_port)
            notebook_server = uvicorn.Server(
                uvicorn.Config(notebook_app, host=notebook_host, port=notebook_port, reload=False)
            )
            tasks.append(
                asyncio.create_task(
                    _run_component("notebook_api", notebook_server.serve(), stop_event, component_errors)
                )
            )

        if session_enabled:
            session_app = create_session_messenger_app(
                gateway=guichet,
                request_timeout_s=session_request_timeout_s,
            )
            logger.info(
                "Starting Session Messenger API on http://%s:%d/session/message",
                session_host,
                session_port,
            )
            session_server = uvicorn.Server(
                uvicorn.Config(session_app, host=session_host, port=session_port, reload=False)
            )
            tasks.append(
                asyncio.create_task(
                    _run_component("session_messenger_api", session_server.serve(), stop_event, component_errors)
                )
            )

        if tui_enabled:
            tui_app = create_tui_app(
                gateway=guichet,
                request_timeout_s=tui_request_timeout_s,
            )
            logger.info("Starting TUI API on http://%s:%d/tui/message", tui_host, tui_port)
            tui_server = uvicorn.Server(uvicorn.Config(tui_app, host=tui_host, port=tui_port, reload=False))
            tasks.append(
                asyncio.create_task(
                    _run_component("tui_api", tui_server.serve(), stop_event, component_errors)
                )
            )

        await stop_event.wait()
    finally:
        for server in (openwebui_server, notebook_server, session_server, tui_server):
            if server is not None:
                server.should_exit = True

        for task in tasks:
            if not task.done():
                try:
                    await asyncio.wait_for(task, timeout=5.0)
                except asyncio.TimeoutError:
                    task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

        if runtime is not None:
            try:
                await runtime["mcp_client"].disconnect()
            except Exception:
                logger.exception("mcp disconnect failed")

    if component_errors:
        first_name, first_exc = component_errors[0]
        raise RuntimeError(f"{first_name} failed: {first_exc}") from first_exc


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    asyncio.run(main())
