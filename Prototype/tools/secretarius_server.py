import json
import random
import sys
from pathlib import Path
from typing import Any

PROTOCOL_VERSION = "2025-11-25"
SERVER_INFO = {"name": "OracleServer", "version": "1.0.0"}
TOOL_NAME = "ask_oracle"
_EXTERNAL_TOOLS: list[dict[str, Any]] = []
_EXTERNAL_TOOL_NAMES: set[str] = set()
_EXTERNAL_HANDLER = None


def _init_external_secretarius_tools() -> None:
    global _EXTERNAL_TOOLS, _EXTERNAL_TOOL_NAMES, _EXTERNAL_HANDLER
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    try:
        from secretarius_local import mcp_server as secretarius_mcp_server
    except Exception:
        _EXTERNAL_TOOLS = []
        _EXTERNAL_TOOL_NAMES = set()
        _EXTERNAL_HANDLER = None
        return

    _EXTERNAL_HANDLER = secretarius_mcp_server.handle_mcp_message
    response = _EXTERNAL_HANDLER({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}})
    tools = (((response or {}).get("result") or {}).get("tools")) if isinstance(response, dict) else None
    if isinstance(tools, list):
        _EXTERNAL_TOOLS = [tool for tool in tools if isinstance(tool, dict)]
    else:
        _EXTERNAL_TOOLS = []
    _EXTERNAL_TOOL_NAMES = {
        str(tool.get("name"))
        for tool in _EXTERNAL_TOOLS
        if isinstance(tool.get("name"), str) and tool.get("name")
    }

def _send_response(request_id, result):
    response = {"jsonrpc": "2.0", "id": request_id, "result": result}
    sys.stdout.write(json.dumps(response) + "\n")
    sys.stdout.flush()


def _send_error(request_id, code: int, message: str):
    response = {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}
    sys.stdout.write(json.dumps(response) + "\n")
    sys.stdout.flush()


def _handle_initialize(request_id):
    _send_response(
        request_id,
        {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": SERVER_INFO,
        },
    )


def _handle_list_tools(request_id):
    local_tools = [
        {
            "name": TOOL_NAME,
            "description": "Test tool: ask the oracle a yes/no question only when explicitly requested.",
            "inputSchema": {
                "type": "object",
                "properties": {"question": {"type": "string"}},
                "required": ["question"],
                "additionalProperties": False,
            },
        }
    ]
    # Put business tools first and keep oracle last to reduce model bias toward
    # this test tool.
    tools = _EXTERNAL_TOOLS + local_tools
    _send_response(
        request_id,
        {
            "tools": tools
        },
    )


def _handle_call_tool(request_id, params):
    name = (params or {}).get("name")
    arguments = (params or {}).get("arguments") or {}
    if isinstance(name, str) and name in _EXTERNAL_TOOL_NAMES and _EXTERNAL_HANDLER is not None:
        response = _EXTERNAL_HANDLER(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            }
        )
        if isinstance(response, dict):
            error = response.get("error")
            if isinstance(error, dict):
                _send_error(request_id, int(error.get("code", -32000)), str(error.get("message", "Tool error")))
                return
            result = response.get("result")
            if isinstance(result, dict):
                _send_response(request_id, result)
                return
        _send_error(request_id, -32000, "External tool execution error")
        return

    if name != TOOL_NAME:
        _send_response(
            request_id,
            {
                "content": [{"type": "text", "text": f"Unknown tool: {name}"}],
                "isError": True,
            },
        )
        return

    question = str(arguments.get("question", ""))
    answer = random.choice(["OUI", "NON"])
    _send_response(
        request_id,
        {
            "content": [
                {
                    "type": "text",
                    "text": f"The oracle pondered your question: '{question}' and answered: {answer}",
                }
            ],
            "isError": False,
        },
    )


def main():
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue

        method = message.get("method")
        request_id = message.get("id")
        params = message.get("params")

        if request_id is None:
            # Notifications do not require a response.
            continue

        if method == "initialize":
            _handle_initialize(request_id)
        elif method == "tools/list":
            _handle_list_tools(request_id)
        elif method == "tools/call":
            _handle_call_tool(request_id, params)
        elif method == "ping":
            _send_response(request_id, {})
        else:
            _send_error(request_id, -32601, f"Method not found: {method}")


if __name__ == "__main__":
    _init_external_secretarius_tools()
    main()
