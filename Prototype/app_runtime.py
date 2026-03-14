import sys
from pathlib import Path
from typing import Any

import yaml

from adapters.output.llm_ollama import OllamaAdapter
from adapters.output.mcp_client import StdioMCPClient
from core.chef_orchestre import ChefDOrchestre
from core.ports import InputGatewayInterface
from localization import DEFAULT_LOCALE


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _resolve_project_path(project_root: Path, maybe_relative_path: str) -> str:
    path = Path(maybe_relative_path)
    if path.is_absolute():
        return str(path)
    return str((project_root / path).resolve())


async def build_runtime(gateway: InputGatewayInterface, config_path: Path | None = None) -> dict[str, Any]:
    project_root = Path(__file__).parent
    cfg_path = config_path or (project_root / "config.yaml")
    config = load_config(str(cfg_path))

    llm_config = config.get("llm", {})
    locale = str(config.get("locale") or DEFAULT_LOCALE)
    llm_raw_log = llm_config.get("raw_log_file") or "logs/llm_raw.log"
    llm_adapter = OllamaAdapter(
        base_url=llm_config.get("base_url") or "http://localhost:11434",
        model=llm_config.get("model") or "qwen3:0.6b",
        think=bool(llm_config.get("think", False)),
        request_timeout_s=float(llm_config.get("request_timeout_s") or 180.0),
        raw_log_path=_resolve_project_path(project_root, llm_raw_log),
    )

    mcp_servers = config.get("mcp_servers", {})
    mcp_config = mcp_servers.get("secretarius") or mcp_servers.get("oracle", {})
    mcp_log_file = mcp_config.get("log_file") or "logs/mcp_server.log"
    mcp_log_path = _resolve_project_path(project_root, mcp_log_file)
    Path(mcp_log_path).parent.mkdir(parents=True, exist_ok=True)
    mcp_env = {"SECRETARIUS_MCP_LOG": mcp_log_path}
    mcp_env["SECRETARIUS_LOCALE"] = locale
    if mcp_config.get("search_min_score") is not None:
        mcp_env["SECRETARIUS_MILVUS_MIN_SCORE"] = str(mcp_config.get("search_min_score"))
    if mcp_config.get("collection_name"):
        mcp_env["SECRETARIUS_MILVUS_COLLECTION"] = str(mcp_config.get("collection_name"))
    mcp_client = StdioMCPClient(
        command=mcp_config.get("command") or sys.executable,
        args=mcp_config.get("args", ["tools/secretarius_server.py"]),
        cwd=str(project_root),
        env=mcp_env,
    )
    await mcp_client.connect()

    orchestrator = ChefDOrchestre(
        llm=llm_adapter,
        tool_client=mcp_client,
        gateway=gateway,
        locale=locale,
    )

    return {
        "config": config,
        "orchestrator": orchestrator,
        "mcp_client": mcp_client,
    }
