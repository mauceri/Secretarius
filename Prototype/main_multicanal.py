#!/usr/bin/env python3

import asyncio
import logging
from pathlib import Path
from typing import Any

import uvicorn

from adapters.input.guichet_unique import GuichetUnique
from adapters.input.tui_guichet import TUIChannel
from app_runtime import build_runtime, load_config
from memos_api import create_memos_app
from notebook_api import create_notebook_app
from openwebui_api import create_openwebui_app
from session_messenger_api import create_session_messenger_app

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
    memos_cfg = config.get("memos", {})

    journal_file = ui_cfg.get("journal_file") or "logs/guichet.log"
    openwebui_journal_file = webui_cfg.get("journal_file") or "logs/openwebui.log"
    notebook_journal_file = notebook_cfg.get("journal_file") or "logs/notebook.log"
    session_journal_file = session_cfg.get("journal_file") or "logs/session_messenger.log"
    memos_journal_file = memos_cfg.get("journal_file") or "logs/memos.log"
    show_thoughts = bool(ui_cfg.get("show_thoughts", True))
    journal_path = str((project_root / journal_file).resolve())
    openwebui_journal_path = str((project_root / openwebui_journal_file).resolve())
    notebook_journal_path = str((project_root / notebook_journal_file).resolve())
    session_journal_path = str((project_root / session_journal_file).resolve())
    memos_journal_path = str((project_root / memos_journal_file).resolve())

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
    memos_enabled = bool(memos_cfg.get("enabled", False))
    memos_host = memos_cfg.get("host") or "127.0.0.1"
    memos_port = int(memos_cfg.get("port") or 8004)
    memos_request_timeout_s = float(memos_cfg.get("request_timeout_s") or 120.0)
    memos_publish_timeout_s = float(memos_cfg.get("publish_timeout_s") or 30.0)
    memos_base_url = memos_cfg.get("base_url") or "http://127.0.0.1:5230"
    memos_access_token = str(memos_cfg.get("access_token") or "")
    memos_response_visibility = str(memos_cfg.get("response_visibility") or "PRIVATE")
    memos_webhook_token = str(memos_cfg.get("webhook_token") or "")
    memos_ignored_creator = str(memos_cfg.get("ignored_creator") or "")

    guichet = GuichetUnique(
        journal_path=journal_path,
        channel_journal_paths={
            "tui": journal_path,
            "openwebui": openwebui_journal_path,
            "notebook": notebook_journal_path,
            "session_messenger": session_journal_path,
            "memos": memos_journal_path,
        },
    )
    runtime: dict[str, Any] | None = None
    server: uvicorn.Server | None = None
    notebook_server: uvicorn.Server | None = None
    session_server: uvicorn.Server | None = None
    memos_server: uvicorn.Server | None = None
    tasks: list[asyncio.Task] = []
    tui_task: asyncio.Task | None = None
    api_task: asyncio.Task | None = None
    notebook_task: asyncio.Task | None = None
    session_task: asyncio.Task | None = None
    memos_task: asyncio.Task | None = None
    stop_event = asyncio.Event()
    component_errors: list[tuple[str, BaseException]] = []

    try:
        runtime = await build_runtime(guichet, config_path=config_path)
        tui = TUIChannel(guichet=guichet, channel_name="tui", show_thoughts=show_thoughts)
        app = create_openwebui_app(
            gateway=guichet,
            model_id=model_id,
            request_timeout_s=request_timeout_s,
        )
        logger.info("Starting OpenAI-compatible API on http://%s:%d/v1", host, port)
        server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, reload=False))
        if notebook_enabled:
            notebook_app = create_notebook_app(
                gateway=guichet,
                model_id=notebook_model_id,
                request_timeout_s=notebook_request_timeout_s,
            )
            logger.info(
                "Starting Notebook OpenAI-compatible API on http://%s:%d/v1",
                notebook_host,
                notebook_port,
            )
            notebook_server = uvicorn.Server(
                uvicorn.Config(notebook_app, host=notebook_host, port=notebook_port, reload=False)
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
        if memos_enabled:
            memos_app = create_memos_app(
                gateway=guichet,
                memos_base_url=memos_base_url,
                memos_access_token=memos_access_token,
                request_timeout_s=memos_request_timeout_s,
                publish_timeout_s=memos_publish_timeout_s,
                response_visibility=memos_response_visibility,
                webhook_token=memos_webhook_token,
                ignored_creator=memos_ignored_creator,
            )
            logger.info(
                "Starting Memos webhook API on http://%s:%d/memos/webhook",
                memos_host,
                memos_port,
            )
            memos_server = uvicorn.Server(
                uvicorn.Config(memos_app, host=memos_host, port=memos_port, reload=False)
            )
        tui_task = asyncio.create_task(_run_component("tui_channel", tui.run(), stop_event, component_errors))
        api_task = asyncio.create_task(_run_component("openwebui_api", server.serve(), stop_event, component_errors))
        tasks = [tui_task, api_task]
        if notebook_server is not None:
            notebook_task = asyncio.create_task(
                _run_component("notebook_api", notebook_server.serve(), stop_event, component_errors)
            )
            tasks.append(notebook_task)
        if session_server is not None:
            session_task = asyncio.create_task(
                _run_component("session_messenger_api", session_server.serve(), stop_event, component_errors)
            )
            tasks.append(session_task)
        if memos_server is not None:
            memos_task = asyncio.create_task(
                _run_component("memos_api", memos_server.serve(), stop_event, component_errors)
            )
            tasks.append(memos_task)
        await stop_event.wait()
    finally:
        if server is not None:
            server.should_exit = True
        if notebook_server is not None:
            notebook_server.should_exit = True
        if session_server is not None:
            session_server.should_exit = True
        if memos_server is not None:
            memos_server.should_exit = True

        # Prefer graceful API shutdown to avoid lifespan cancellation tracebacks.
        if api_task is not None and not api_task.done():
            try:
                await asyncio.wait_for(api_task, timeout=5.0)
            except asyncio.TimeoutError:
                api_task.cancel()
        if notebook_task is not None and not notebook_task.done():
            try:
                await asyncio.wait_for(notebook_task, timeout=5.0)
            except asyncio.TimeoutError:
                notebook_task.cancel()
        if session_task is not None and not session_task.done():
            try:
                await asyncio.wait_for(session_task, timeout=5.0)
            except asyncio.TimeoutError:
                session_task.cancel()
        if memos_task is not None and not memos_task.done():
            try:
                await asyncio.wait_for(memos_task, timeout=5.0)
            except asyncio.TimeoutError:
                memos_task.cancel()

        # TUI can be cancelled immediately if still running.
        if tui_task is not None and not tui_task.done():
            tui_task.cancel()

        # Cancel any remaining tasks defensively.
        for task in tasks:
            if not task.done():
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
