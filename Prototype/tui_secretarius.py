#!/usr/bin/env python3

import asyncio
from pathlib import Path

from adapters.input.tui_guichet import RemoteTUIChannel
from app_runtime import load_config


async def main() -> None:
    project_root = Path(__file__).parent
    config = load_config(str(project_root / "config.yaml"))
    ui_cfg = config.get("ui", {})
    tui_cfg = config.get("tui_api", {})
    show_thoughts = bool(ui_cfg.get("show_thoughts", True))
    host = tui_cfg.get("host") or "127.0.0.1"
    port = int(tui_cfg.get("port") or 8003)

    channel = RemoteTUIChannel(
        api_base_url=f"http://{host}:{port}",
        show_thoughts=show_thoughts,
    )
    await channel.run()


if __name__ == "__main__":
    asyncio.run(main())
