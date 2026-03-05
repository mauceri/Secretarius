import asyncio
import json
import logging
import re

from .models import Message, Role, SessionState, Priority, ContextItem
from .ports import LLMInterface, ToolClientInterface, InputGatewayInterface

logger = logging.getLogger(__name__)

REACT_SYSTEM_PROMPT = """
You are Secretarius, a helpful AI agent.
You operate in a Think -> Action -> Observation cycle.
You have access to the following tools:
{tools_schema}

Tool selection rules:
- Choose the most relevant domain tool for the user request.
- Use "ask_oracle" only if the user explicitly asks to consult the oracle.
- Do not use "ask_oracle" as a default fallback when other tools apply.

You MUST respond ONLY with a valid JSON object matching this schema:
{{
    "action": "Name of the tool to use, or null if no tool is needed",
    "action_input": {{"arg1": "value1"}}
}}
Do not provide final_answer.
Do not output chain-of-thought or hidden reasoning.
"""

NO_TOOL_FALLBACK_MESSAGE = "Aucun outil ne correspond a votre demande."
MAX_TOOL_CALLS_PER_TURN = 2

class ChefDOrchestre:
    def __init__(
        self,
        llm: LLMInterface,
        tool_client: ToolClientInterface,
        gateway: InputGatewayInterface
    ):
        self.llm = llm
        self.tool_client = tool_client
        self.gateway = gateway
        self.state = SessionState()
        self._cycle_lock = asyncio.Lock()
        
        # Wire the gateway callback to our input handler
        self.gateway.set_callback(self.handle_user_input)

    async def _display_thought(self, thought: str, phase: str) -> None:
        try:
            await self.gateway.display_thought(thought)
        except Exception:
            logger.exception("UI display_thought failed (phase=%s)", phase)

    async def _display_message(self, role: str, content: str, phase: str) -> None:
        try:
            await self.gateway.display_message(role, content)
        except Exception:
            logger.exception("UI display_message failed (phase=%s, role=%s)", phase, role)

    def _trim_state(self):
        # Keep memory bounded to avoid context bloat over long sessions.
        max_messages = 100
        max_context_items = 200
        if len(self.state.messages) > max_messages:
            self.state.messages = self.state.messages[-max_messages:]
        if len(self.state.context_items) > max_context_items:
            self.state.context_items = self.state.context_items[-max_context_items:]

    def _build_context_prompt(self, include_session_history: bool = True) -> str:
        # Priority 1: Tool Results
        # Priority 2: Session History
        # Priority 3: Semantic Memory
        
        p1 = [c for c in self.state.context_items if c.priority == Priority.TOOL_RESULT]
        p2 = [c for c in self.state.context_items if c.priority == Priority.SESSION_HISTORY]
        p3 = [c for c in self.state.context_items if c.priority == Priority.SEMANTIC_MEMORY]
        
        context_str = "CONTEXT:\n"
        if p1:
            context_str += "--- Priority 1: Tool Results ---\n"
            for c in p1[-10:]: # keep recent
                context_str += f"{c.content}\n"
        if include_session_history and p2:
            context_str += "--- Priority 2: Session History ---\n"
            for c in p2[-20:]:
                context_str += f"{c.content}\n"
        if p3:
            context_str += "--- Priority 3: Semantic Memory ---\n"
            for c in p3:
                context_str += f"{c.content}\n"
            
        return context_str

    def _build_router_tools_schema(self, tools: list[dict]) -> str:
        compact_tools: list[dict] = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            name = str(tool.get("name") or "").strip()
            if not name:
                continue
            description = str(tool.get("description") or "").strip()
            input_schema = tool.get("inputSchema") if isinstance(tool.get("inputSchema"), dict) else {}
            properties = input_schema.get("properties") if isinstance(input_schema.get("properties"), dict) else {}
            required = input_schema.get("required") if isinstance(input_schema.get("required"), list) else []
            fields = sorted(str(k) for k in properties.keys())
            compact_tools.append(
                {
                    "name": name,
                    "description": description[:220],
                    "input_fields": fields,
                    "required": [str(k) for k in required],
                }
            )
        return json.dumps(compact_tools, ensure_ascii=False, indent=2)

    def _sanitize_action_input(self, action: str, action_input: dict) -> dict:
        if not isinstance(action_input, dict):
            return {}

        if action == "ask_oracle":
            question = action_input.get("question")
            if question is None:
                return {}
            return {"question": str(question)}

        if action == "extract_expressions":
            allowed = {"text", "document"}
            return {k: v for k, v in action_input.items() if k in allowed}

        if action == "expressions_to_embeddings":
            allowed = {"expressions", "document", "model", "normalize", "batch_size"}
            return {k: v for k, v in action_input.items() if k in allowed}

        if action == "semantic_graph_search":
            allowed = {
                "embeddings",
                "expressions",
                "documents",
                "top_k",
                "document",
                "upsert",
                "milvus_uri",
                "milvus_token",
                "collection_name",
                "metric_type",
                "model",
                "normalize",
                "batch_size",
            }
            return {k: v for k, v in action_input.items() if k in allowed}

        if action == "index_text":
            allowed = {
                "text",
                "document",
                "collection_name",
                "milvus_uri",
                "milvus_token",
                "metric_type",
                "model",
                "normalize",
                "batch_size",
            }
            return {k: v for k, v in action_input.items() if k in allowed}

        if action == "search_text":
            allowed = {
                "query",
                "top_k",
                "collection_name",
                "milvus_uri",
                "milvus_token",
                "metric_type",
                "model",
                "normalize",
                "batch_size",
            }
            return {k: v for k, v in action_input.items() if k in allowed}

        return action_input

    def _is_explicit_oracle_request(self) -> bool:
        user_text = ""
        for msg in reversed(self.state.messages):
            if msg.role == Role.USER:
                user_text = (msg.content or "").strip().lower()
                if user_text:
                    break
        if not user_text:
            return False

        # Keep oracle tool opt-in strict to avoid random fallback on unrelated requests.
        oracle_markers = (
            "oracle",
            "orakel",
            "proph",
            "divination",
            "voyance",
            "boule de cristal",
            "ask_oracle",
        )
        return any(marker in user_text for marker in oracle_markers)

    def _latest_user_text(self) -> str:
        for msg in reversed(self.state.messages):
            if msg.role == Role.USER and isinstance(msg.content, str):
                text = msg.content.strip()
                if text:
                    return text
        return ""

    def _ensure_extract_expressions_payload(self, action_input: dict) -> dict:
        if not isinstance(action_input, dict):
            action_input = {}

        text = action_input.get("text")
        has_text = isinstance(text, str) and bool(text.strip())

        document = action_input.get("document")
        has_document_text = False
        if isinstance(document, dict):
            content = document.get("content")
            if isinstance(content, dict):
                doc_text = content.get("text")
                has_document_text = isinstance(doc_text, str) and bool(doc_text.strip())

        if has_text or has_document_text:
            return action_input

        fallback_text = self._latest_user_text()
        if fallback_text:
            action_input["text"] = fallback_text
            return action_input

        return action_input

    @staticmethod
    def _strip_extract_instruction_prefix(text: str) -> str:
        if not isinstance(text, str):
            return text
        t = text.strip()
        if not t:
            return t

        # Remove leading imperative prompt wrapper while preserving poem/content body.
        # Examples:
        # "extraire les expressions caractéristiques de : <body>"
        # "peux-tu extraire ... ? <body>"
        patterns = [
            r"(?is)^\s*(?:\*+\s*)?(?:peux-tu|pouvez-vous|merci de|veuillez)?\s*"
            r"(?:extraire|extrait|extrais)\b.*?(?::|\n)\s*(.+)$",
            r"(?is)^\s*(?:\*+\s*)?(?:extraire|extrait|extrais)\b.*?\?\s*(.+)$",
        ]
        for pattern in patterns:
            m = re.match(pattern, t)
            if m:
                body = (m.group(1) or "").strip()
                if body:
                    return body
        return t

    async def _execute_react_cycle(self):
        tools = await self.tool_client.list_tools()
        tool_names = {t.get("name") for t in tools if isinstance(t, dict)}
        tools_schema = self._build_router_tools_schema(tools)
        system_prompt = REACT_SYSTEM_PROMPT.format(tools_schema=tools_schema)
        # Router mode: avoid duplicating the current user message in both
        # `messages` and context prompt.
        system_prompt += "\n" + self._build_context_prompt(include_session_history=False)

        called_actions_in_cycle: set[str] = set()
        for _ in range(MAX_TOOL_CALLS_PER_TURN):
            try:
                raw_response = await self.llm.generate_response(self.state.messages, system_prompt)

                # Try parsing JSON
                try:
                    # Some LLMs wrap JSON in markdown blocks
                    clean_response = raw_response.strip()
                    if clean_response.startswith('```json'):
                        clean_response = clean_response[7:]
                    elif clean_response.startswith('```'):
                        clean_response = clean_response[3:]
                    if clean_response.endswith('```'):
                        clean_response = clean_response[:-3]
                    clean_response = clean_response.strip()
                        
                    response_data = json.loads(clean_response)
                except json.JSONDecodeError:
                    await self._display_thought(
                        f"Failed to parse LLM JSON: {raw_response}",
                        phase="parse_json_error",
                    )
                    await self._display_message("Secretarius", NO_TOOL_FALLBACK_MESSAGE, phase="no_tool_json_error")
                    self.state.messages.append(Message(role=Role.ASSISTANT, content=NO_TOOL_FALLBACK_MESSAGE))
                    self.state.context_items.append(
                        ContextItem(priority=Priority.SESSION_HISTORY, content=f"Assistant: {NO_TOOL_FALLBACK_MESSAGE}")
                    )
                    self._trim_state()
                    return

                logger.info(f"############################################# {response_data}")
                action = response_data.get("action")
                if isinstance(action, str):
                    action = action.strip() or None
                action_input = response_data.get("action_input", {})
                if not isinstance(action_input, dict):
                    action_input = {}
                if response_data.get("final_answer") is not None:
                    await self._display_thought(
                        "Ignored model final_answer: router mode requires tool-only decisions.",
                        phase="ignored_model_final_answer",
                    )

                if not action or action not in tool_names:
                    await self._display_message("Secretarius", NO_TOOL_FALLBACK_MESSAGE, phase="no_matching_tool")
                    self.state.messages.append(Message(role=Role.ASSISTANT, content=NO_TOOL_FALLBACK_MESSAGE))
                    self.state.context_items.append(
                        ContextItem(priority=Priority.SESSION_HISTORY, content=f"Assistant: {NO_TOOL_FALLBACK_MESSAGE}")
                    )
                    self._trim_state()
                    return

                if action == "ask_oracle" and not self._is_explicit_oracle_request():
                    await self._display_thought(
                        "Blocked ask_oracle: user did not explicitly request oracle usage.",
                        phase="oracle_policy_block",
                    )
                    await self._display_message("Secretarius", NO_TOOL_FALLBACK_MESSAGE, phase="oracle_policy_block_stop")
                    self.state.messages.append(Message(role=Role.ASSISTANT, content=NO_TOOL_FALLBACK_MESSAGE))
                    self.state.context_items.append(
                        ContextItem(priority=Priority.SESSION_HISTORY, content=f"Assistant: {NO_TOOL_FALLBACK_MESSAGE}")
                    )
                    self._trim_state()
                    return

                sanitized_action_input = self._sanitize_action_input(action, action_input)
                if action == "extract_expressions":
                    sanitized_action_input = self._ensure_extract_expressions_payload(sanitized_action_input)
                    text_value = sanitized_action_input.get("text")
                    if isinstance(text_value, str) and text_value.strip():
                        sanitized_action_input["text"] = self._strip_extract_instruction_prefix(text_value)
                removed_keys = sorted(set(action_input.keys()) - set(sanitized_action_input.keys()))
                if removed_keys:
                    await self._display_thought(
                        f"Dropped unsupported tool args for '{action}': {removed_keys}",
                        phase="tool_input_sanitized",
                    )
                action_input = sanitized_action_input
                action_signature = f"{action}:{json.dumps(action_input, sort_keys=True, ensure_ascii=False)}"
                if action_signature in called_actions_in_cycle:
                    await self._display_message("Secretarius", NO_TOOL_FALLBACK_MESSAGE, phase="duplicate_action_guard")
                    self.state.messages.append(Message(role=Role.ASSISTANT, content=NO_TOOL_FALLBACK_MESSAGE))
                    self.state.context_items.append(
                        ContextItem(priority=Priority.SESSION_HISTORY, content=f"Assistant: {NO_TOOL_FALLBACK_MESSAGE}")
                    )
                    self._trim_state()
                    return
                called_actions_in_cycle.add(action_signature)

                await self._display_thought(
                    f"Calling tool: {action} with inputs {action_input}",
                    phase="tool_call_start",
                )
                try:
                    observation = await self.tool_client.call_tool(action, action_input)
                except Exception as e:
                    observation = f"Error calling tool {action}: {e}"

                await self._display_thought(
                    f"Observation: {observation}",
                    phase="tool_call_observation",
                )

                tool_context = f"Tool '{action}' returned: {observation}"
                self.state.context_items.append(ContextItem(
                    priority=Priority.TOOL_RESULT,
                    content=tool_context
                ))

                self.state.messages.append(Message(
                    role=Role.TOOL,
                    name=action,
                    content=observation
                ))

                # Router mode: return tool output directly, no post-tool LLM formatting.
                tool_output = str(observation).strip()
                self.state.messages.append(Message(role=Role.ASSISTANT, content=tool_output))
                await self._display_message("Secretarius", tool_output, phase="final_answer_from_tool")
                self.state.context_items.append(
                    ContextItem(priority=Priority.SESSION_HISTORY, content=f"Assistant: {tool_output}")
                )
                self._trim_state()
                return
            except Exception as e:
                logger.error(f"Error in ReAct cycle: {e}")
                await self._display_message(
                    "System",
                    f"An internal error occurred: {e}",
                    phase="react_cycle_exception",
                )
                return

        await self._display_message("Secretarius", NO_TOOL_FALLBACK_MESSAGE, phase="max_tool_calls_reached")
        self.state.messages.append(Message(role=Role.ASSISTANT, content=NO_TOOL_FALLBACK_MESSAGE))
        self.state.context_items.append(
            ContextItem(priority=Priority.SESSION_HISTORY, content=f"Assistant: {NO_TOOL_FALLBACK_MESSAGE}")
        )
        self._trim_state()

    async def handle_user_input(self, user_input: str):
        async with self._cycle_lock:
            # Stateless-by-default: reset state for every user request,
            # regardless of channel, to avoid anchoring on prior turns.
            self.state = SessionState()

            # 1. Update State
            self.state.messages.append(Message(role=Role.USER, content=user_input))
            self.state.context_items.append(ContextItem(
                priority=Priority.SESSION_HISTORY,
                content=f"User: {user_input}"
            ))
            self._trim_state()
            
            # 2. Run ReAct loop
            await self._execute_react_cycle()
