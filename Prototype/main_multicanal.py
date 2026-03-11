#!/usr/bin/env python3

import asyncio
import logging
from pathlib import Path
from typing import Any

import uvicorn

from adapters.input.guichet_unique import GuichetUnique
from adapters.input.tui_guichet import TUIChannel
from app_runtime import build_runtime, load_config
from notebook_api import create_notebook_app
from openwebui_api import create_openwebui_app

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

    journal_file = ui_cfg.get("journal_file") or "logs/guichet.log"
    openwebui_journal_file = webui_cfg.get("journal_file") or "logs/openwebui.log"
    notebook_journal_file = notebook_cfg.get("journal_file") or "logs/notebook.log"
    show_thoughts = bool(ui_cfg.get("show_thoughts", True))
    journal_path = str((project_root / journal_file).resolve())
    openwebui_journal_path = str((project_root / openwebui_journal_file).resolve())
    notebook_journal_path = str((project_root / notebook_journal_file).resolve())

    host = webui_cfg.get("host") or "0.0.0.0"
    port = int(webui_cfg.get("port") or 8000)
    model_id = webui_cfg.get("model_id") or "secretarius-agent"
    request_timeout_s = float(webui_cfg.get("request_timeout_s") or 90.0)
    notebook_enabled = bool(notebook_cfg.get("enabled", True))
    notebook_host = notebook_cfg.get("host") or "0.0.0.0"
    notebook_port = int(notebook_cfg.get("port") or 8001)
    notebook_model_id = notebook_cfg.get("model_id") or "secretarius-notebook"
    notebook_request_timeout_s = float(notebook_cfg.get("request_timeout_s") or 120.0)

    guichet = GuichetUnique(
        journal_path=journal_path,
        channel_journal_paths={
            "tui": journal_path,
            "openwebui": openwebui_journal_path,
            "notebook": notebook_journal_path,
        },
    )
    runtime: dict[str, Any] | None = None
    server: uvicorn.Server | None = None
    notebook_server: uvicorn.Server | None = None
    tasks: list[asyncio.Task] = []
    tui_task: asyncio.Task | None = None
    api_task: asyncio.Task | None = None
    notebook_task: asyncio.Task | None = None
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
        tui_task = asyncio.create_task(_run_component("tui_channel", tui.run(), stop_event, component_errors))
        api_task = asyncio.create_task(_run_component("openwebui_api", server.serve(), stop_event, component_errors))
        tasks = [tui_task, api_task]
        if notebook_server is not None:
            notebook_task = asyncio.create_task(
                _run_component("notebook_api", notebook_server.serve(), stop_event, component_errors)
            )
            tasks.append(notebook_task)
        await stop_event.wait()
    finally:
        if server is not None:
            server.should_exit = True
        if notebook_server is not None:
            notebook_server.should_exit = True

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
