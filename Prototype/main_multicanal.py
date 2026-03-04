#!/usr/bin/env python3

import asyncio
import logging
from pathlib import Path
import socket
from typing import Any

import uvicorn

from adapters.input.guichet_unique import GuichetUnique
from adapters.input.tui_guichet import TUIChannel
from app_runtime import build_runtime, load_config
from openwebui_api import create_openwebui_app

logger = logging.getLogger(__name__)


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
                logger.warning(
                    "openwebui port %d unavailable, using fallback port %d",
                    preferred_port,
                    candidate,
                )
            return candidate
    raise RuntimeError(
        f"No available OpenWebUI port in range {preferred_port}-{preferred_port + max_attempts - 1}"
    )


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

    journal_file = ui_cfg.get("journal_file") or "logs/guichet.log"
    journal_path = str((project_root / journal_file).resolve())

    host = webui_cfg.get("host") or "0.0.0.0"
    preferred_port = int(webui_cfg.get("port") or 8000)
    port_fallback_attempts = int(webui_cfg.get("port_fallback_attempts") or 3)
    port = _pick_openwebui_port(host, preferred_port, port_fallback_attempts)
    model_id = webui_cfg.get("model_id") or "secretarius-agent"
    request_timeout_s = float(webui_cfg.get("request_timeout_s") or 90.0)

    guichet = GuichetUnique(journal_path=journal_path)
    runtime: dict[str, Any] | None = None
    server: uvicorn.Server | None = None
    tasks: list[asyncio.Task] = []
    stop_event = asyncio.Event()
    component_errors: list[tuple[str, BaseException]] = []

    try:
        runtime = await build_runtime(guichet, config_path=config_path)
        tui = TUIChannel(guichet=guichet, channel_name="tui")
        app = create_openwebui_app(
            gateway=guichet,
            model_id=model_id,
            request_timeout_s=request_timeout_s,
        )
        server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, reload=False))
        tasks = [
            asyncio.create_task(_run_component("tui_channel", tui.run(), stop_event, component_errors)),
            asyncio.create_task(_run_component("openwebui_api", server.serve(), stop_event, component_errors)),
        ]
        await stop_event.wait()
    finally:
        if server is not None:
            server.should_exit = True
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
