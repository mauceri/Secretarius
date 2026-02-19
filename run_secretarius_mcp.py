from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_run_stdio_server():
    module_path = Path(__file__).resolve().parent / "secretarius" / "mcp_server.py"
    spec = importlib.util.spec_from_file_location("secretarius_mcp_server", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load MCP server module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.run_stdio_server


if __name__ == "__main__":
    run_stdio_server = _load_run_stdio_server()
    run_stdio_server()
