import asyncio
import logging
from pathlib import Path

from adapters.input.guichet_unique import GuichetUnique
from adapters.input.tui_guichet import TUIChannel
from app_runtime import build_runtime, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# To suppress httpx info logs
logging.getLogger("httpx").setLevel(logging.WARNING)

async def main():
    config_path = Path(__file__).parent / "config.yaml"
    config = load_config(str(config_path))

    # Single guichet shared by all channels in this process.
    ui_config = config.get("ui", {})
    journal_file = ui_config.get("journal_file") or "logs/guichet.log"
    journal_path = str((Path(__file__).parent / journal_file).resolve())
    guichet = GuichetUnique(journal_path=journal_path)
    tui = TUIChannel(guichet=guichet, channel_name="tui")

    try:
        runtime = await build_runtime(guichet, config_path=config_path)
    except Exception as e:
        logger.error(f"Failed to start/connect to MCP client: {e}")
        return
    mcp_client = runtime["mcp_client"]

    # Keep reference explicit for readability/debugging.
    orchestrator = runtime["orchestrator"]
    _ = orchestrator

    # Run Gateway
    try:
        await tui.run()
    finally:
        # Cleanup
        await mcp_client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
