from __future__ import annotations

import logging
import os
import json
import re
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

LOGGER = logging.getLogger("secretarius.agent")
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


@dataclass(frozen=True)
class AgentConfig:
    model: str
    model_server: str
    api_key: str
    python_bin: str
    mcp_entry: str
    include_code_interpreter: bool
    thought_in_content: bool
    think: bool

    @staticmethod
    def from_env() -> "AgentConfig":
        project_root = Path(__file__).resolve().parent.parent
        python_default = str(project_root / ".venv" / "bin" / "python")
        mcp_default = str(project_root / "run_secretarius_mcp.py")
        return AgentConfig(
#            model=os.environ.get("SECRETARIUS_QWEN_MODEL", "Qwen3-0.6B"),
            model=os.environ.get("SECRETARIUS_QWEN_MODEL", "Qwen3-0.6B"),
            model_server=os.environ.get("SECRETARIUS_QWEN_MODEL_SERVER", "http://127.0.0.1:8000/v1"),
            api_key=os.environ.get("SECRETARIUS_QWEN_API_KEY", "EMPTY"),
            python_bin=os.environ.get("SECRETARIUS_MCP_PYTHON", python_default),
            mcp_entry=os.environ.get("SECRETARIUS_MCP_ENTRYPOINT", mcp_default),
            include_code_interpreter=(
                os.environ.get("SECRETARIUS_QWEN_CODE_INTERPRETER", "false").strip().lower()
                in ("1", "true", "yes", "on")
            ),
            thought_in_content=(
                os.environ.get("SECRETARIUS_QWEN_THOUGHT_IN_CONTENT", "false").strip().lower()
                in ("1", "true", "yes", "on")
            ),
            think=(
                os.environ.get("SECRETARIUS_QWEN_THINK", "false").strip().lower()
                in ("1", "true", "yes", "on")
            ),
        )


class SecretariusAgentRuntime:
    def __init__(self, config: AgentConfig | None = None) -> None:
        self.config = config or AgentConfig.from_env()
        self._bot: Any | None = None
        self._lock = Lock()

    def _build_tools(self) -> list[Any]:
        tools: list[Any] = [
            {
                "mcpServers": {
                    "secretarius": {
                        "command": self.config.python_bin,
                        "args": ["-u", self.config.mcp_entry],
                    }
                }
            }
        ]
        if self.config.include_code_interpreter:
            tools.append("code_interpreter")
        return tools

    def _ensure_bot(self) -> Any:
        if self._bot is not None:
            return self._bot
        with self._lock:
            if self._bot is not None:
                return self._bot
            try:
                from qwen_agent.agents import Assistant
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("qwen-agent is not installed in this environment") from exc

            llm_cfg = {
                "model": self.config.model,
                "model_server": self.config.model_server,
                "api_key": self.config.api_key,
                "generate_cfg": {
                    "thought_in_content": self.config.thought_in_content,
                    "extra_body": {
                        "think": self.config.think,
                    },
                },
            }
            self._bot = Assistant(llm=llm_cfg, function_list=self._build_tools())
            return self._bot

    def run(self, messages: list[dict[str, str]]) -> str:
        bot = self._ensure_bot()
        last_response: Any = None
        LOGGER.info("********************************************messages: %s", messages)
        for responses in bot.run(messages=messages):
            #LOGGER.info("********************************************responses: %s", responses)
            last_response = responses
            if self._contains_function_call(responses):
                LOGGER.info("********************************************break_on_first_function_call")
                break
        LOGGER.info("********************************************last_response: %s", last_response)

        # First, handle native qwen-agent function_call payloads directly.
        direct_tool_payload = self._maybe_execute_function_call_from_response(last_response)
        if direct_tool_payload is not None:
            return direct_tool_payload

        output_text = self._extract_text(last_response)
        LOGGER.info("********************************************output_text: %s", output_text)
        tool_payload = self._maybe_execute_tool_call(output_text)
        if tool_payload is not None:
            return tool_payload
        extracted_json = _extract_first_json_object(output_text)
        if extracted_json is not None:
            return json.dumps(extracted_json, ensure_ascii=False)
        return output_text

    @staticmethod
    def _extract_text(last_response: Any) -> str:
        if last_response is None:
            return ""
        if isinstance(last_response, str):
            return last_response.strip()
        if isinstance(last_response, list):
            parts: list[str] = []
            for item in last_response:
                if isinstance(item, dict):
                    content = item.get("content")
                    if isinstance(content, str) and content.strip():
                        parts.append(content.strip())
            return "\n".join(parts).strip()
        if isinstance(last_response, dict):
            content = last_response.get("content")
            if isinstance(content, str):
                return content.strip()
        return str(last_response).strip()

    @staticmethod
    def _maybe_execute_tool_call(output_text: str) -> str | None:
        raw = _extract_tool_call_json(output_text)
        if not isinstance(raw, dict):
            return None
        name = raw.get("name")
        arguments = raw.get("arguments", {})
        if not isinstance(arguments, dict):
            # Accept common alternative shape used in some prompts.
            alt = raw.get("parameters")
            if isinstance(alt, dict):
                arguments = alt
        if not isinstance(name, str) or not isinstance(arguments, dict):
            return None

        tool_name = _canonical_tool_name(name)
        try:
            from .mcp_server import _handle_tool_call as handle_tool_call_local
            result = handle_tool_call_local(tool_name, arguments)
            payload = result.get("structuredContent")
            if isinstance(payload, dict):
                return json.dumps(payload, ensure_ascii=False)
            return json.dumps(result, ensure_ascii=False)
        except Exception as exc:
            return json.dumps(
                {"status": "error", "tool": tool_name, "error": str(exc)},
                ensure_ascii=False,
            )

    @staticmethod
    def _maybe_execute_function_call_from_response(last_response: Any) -> str | None:
        if not isinstance(last_response, list):
            return None

        for item in last_response:
            if not isinstance(item, dict):
                continue
            fc = item.get("function_call")
            if not isinstance(fc, dict):
                continue
            name = fc.get("name")
            raw_args = fc.get("arguments")
            if not isinstance(name, str):
                continue

            arguments: dict[str, Any] = {}
            if isinstance(raw_args, dict):
                arguments = raw_args
            elif isinstance(raw_args, str) and raw_args.strip():
                try:
                    parsed = json.loads(raw_args)
                    if isinstance(parsed, dict):
                        arguments = parsed
                except (TypeError, ValueError, json.JSONDecodeError):
                    arguments = {}

            # Accept wrapped payloads from some model outputs:
            # {"parameters": {...}} or {"arguments": {...}}
            if isinstance(arguments.get("parameters"), dict):
                arguments = arguments["parameters"]
            elif isinstance(arguments.get("arguments"), dict):
                arguments = arguments["arguments"]

            tool_name = _canonical_tool_name(name)
            try:
                from .mcp_server import _handle_tool_call as handle_tool_call_local
                result = handle_tool_call_local(tool_name, arguments)
                payload = result.get("structuredContent")
                if isinstance(payload, dict):
                    return json.dumps(payload, ensure_ascii=False)
                return json.dumps(result, ensure_ascii=False)
            except Exception as exc:
                return json.dumps(
                    {"status": "error", "tool": tool_name, "error": str(exc)},
                    ensure_ascii=False,
                )
        return None

    @staticmethod
    def _contains_function_call(response: Any) -> bool:
        if isinstance(response, dict):
            return isinstance(response.get("function_call"), dict)
        if isinstance(response, list):
            for item in response:
                if isinstance(item, dict) and isinstance(item.get("function_call"), dict):
                    return True
        return False


def _extract_tool_call_json(text: str) -> dict[str, Any] | None:
    src = text or ""
    match = TOOL_CALL_RE.search(src)
    if match:
        try:
            obj = json.loads(match.group(1))
            if isinstance(obj, dict) and "name" in obj and "arguments" in obj:
                return obj
        except (TypeError, ValueError, json.JSONDecodeError):
            pass

    decoder = json.JSONDecoder()
    for i, ch in enumerate(src):
        if ch != "{":
            continue
        try:
            obj, _end = decoder.raw_decode(src[i:])
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(obj, dict) and "name" in obj and "arguments" in obj:
            return obj
    return None


def _canonical_tool_name(name: str) -> str:
    tool_name = (name or "").strip().replace("-", "_")
    if tool_name.startswith("secretarius_"):
        tool_name = tool_name[len("secretarius_") :]
    return tool_name


def _extract_first_json_object(text: str) -> dict[str, Any] | None:
    src = text or ""
    decoder = json.JSONDecoder()
    for i, ch in enumerate(src):
        if ch != "{":
            continue
        try:
            obj, _end = decoder.raw_decode(src[i:])
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(obj, dict):
            return obj
    return None


_RUNTIME: SecretariusAgentRuntime | None = None
_RUNTIME_LOCK = Lock()


def get_runtime() -> SecretariusAgentRuntime:
    global _RUNTIME
    if _RUNTIME is not None:
        return _RUNTIME
    with _RUNTIME_LOCK:
        if _RUNTIME is None:
            _RUNTIME = SecretariusAgentRuntime()
    return _RUNTIME
