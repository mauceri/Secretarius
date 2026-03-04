import asyncio
import json
import logging

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
    "thought": "Your reasoning behind what to do next",
    "action": "Name of the tool to use, or null if no action is needed",
    "action_input": {{"arg1": "value1"}},
    "final_answer": "Your response to the user if no further action is needed (mutually exclusive with action)"
}}
"""

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

    def _build_context_prompt(self) -> str:
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
        if p2:
            context_str += "--- Priority 2: Session History ---\n"
            for c in p2[-20:]:
                context_str += f"{c.content}\n"
        if p3:
            context_str += "--- Priority 3: Semantic Memory ---\n"
            for c in p3:
                context_str += f"{c.content}\n"
            
        return context_str

    async def _execute_react_cycle(self):
        tools = await self.tool_client.list_tools()
        tool_names = {t.get("name") for t in tools if isinstance(t, dict)}
        tools_schema = json.dumps(tools, indent=2)
        system_prompt = REACT_SYSTEM_PROMPT.format(tools_schema=tools_schema)
        system_prompt += "\n" + self._build_context_prompt()

        max_steps = 10
        invalid_response_count = 0
        called_actions_in_cycle: set[str] = set()
        for step in range(max_steps):
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
                    self.state.messages.append(Message(
                        role=Role.USER, 
                        content="Your previous response was not valid JSON. Please respond ONLY with a raw JSON object matching the schema."
                    ))
                    self._trim_state()
                    continue

                thought = response_data.get("thought", "")
                if thought:
                    await self._display_thought(thought, phase="thought")

                action = response_data.get("action")
                if isinstance(action, str):
                    action = action.strip() or None
                action_input = response_data.get("action_input", {})
                final_answer = response_data.get("final_answer")
                if isinstance(final_answer, str):
                    final_answer = final_answer.strip() or None

                if final_answer and action:
                    # Tolerance strategy for small models:
                    # - before any tool call in this cycle, if action is valid, execute it.
                    # - after at least one tool call, favor finalization to avoid duplicated calls.
                    if action in tool_names and not called_actions_in_cycle:
                        await self._display_thought(
                            f"Model returned both action and final_answer; proceeding with action '{action}'.",
                            phase="validation_recovered_prefer_action",
                        )
                        final_answer = None
                    elif called_actions_in_cycle and action in tool_names:
                        await self._display_thought(
                            "Model returned both action and final_answer after a tool call; proceeding with final_answer.",
                            phase="validation_recovered_prefer_final_answer",
                        )
                        action = None
                    else:
                        invalid_response_count += 1
                        await self._display_thought(
                            "Invalid response: both action and final_answer were provided.",
                            phase="validation_error",
                        )
                        self.state.messages.append(Message(
                            role=Role.USER,
                            content=(
                                "Your previous response included both action and final_answer. "
                                "Choose exactly one."
                            )
                        ))
                        self._trim_state()
                        if invalid_response_count >= 3:
                            await self._display_message(
                                "System",
                                "Le modèle renvoie des réponses invalides répétées. Réessayez avec une formulation plus simple.",
                                phase="too_many_invalid_responses",
                            )
                            break
                        continue

                if final_answer:
                    invalid_response_count = 0
                    # Complete cycle
                    self.state.messages.append(Message(role=Role.ASSISTANT, content=final_answer))
                    await self._display_message("Secretarius", final_answer, phase="final_answer")
                    
                    self.state.context_items.append(
                        ContextItem(priority=Priority.SESSION_HISTORY, content=f"Assistant: {final_answer}")
                    )
                    self._trim_state()
                    break

                if action:
                    invalid_response_count = 0
                    action_signature = f"{action}:{json.dumps(action_input, sort_keys=True, ensure_ascii=False)}"
                    if action_signature in called_actions_in_cycle:
                        await self._display_thought(
                            f"Action '{action}' with same inputs already called in this cycle; requesting final_answer.",
                            phase="duplicate_action_guard",
                        )
                        self.state.messages.append(Message(
                            role=Role.USER,
                            content=(
                                "You already called this tool with the same inputs in this cycle and got an observation. "
                                "Do not call the tool again. Provide final_answer now."
                            ),
                        ))
                        self._trim_state()
                        invalid_response_count += 1
                        if invalid_response_count >= 3:
                            await self._display_message(
                                "System",
                                "Le modèle répète les mêmes appels outil. Réessayez avec une formulation plus simple.",
                                phase="duplicate_action_guard_stop",
                            )
                            break
                        continue
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

                    # Priorité 1 : Tool result
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
                    self._trim_state()
                    continue

                # If neither action nor final answer is returned, force a correction.
                invalid_response_count += 1
                self.state.messages.append(Message(
                    role=Role.USER,
                    content=(
                        "Your previous response had neither action nor final_answer. "
                        "Provide exactly one of them."
                    )
                ))
                self._trim_state()
                if invalid_response_count >= 3:
                    await self._display_message(
                        "System",
                        "Le modèle renvoie des réponses invalides répétées. Réessayez avec une formulation plus simple.",
                        phase="too_many_invalid_responses",
                    )
                    break
                    
            except Exception as e:
                logger.error(f"Error in ReAct cycle: {e}")
                await self._display_message(
                    "System",
                    f"An internal error occurred: {e}",
                    phase="react_cycle_exception",
                )
                break

    async def handle_user_input(self, user_input: str):
        async with self._cycle_lock:
            # 1. Update State
            self.state.messages.append(Message(role=Role.USER, content=user_input))
            self.state.context_items.append(ContextItem(
                priority=Priority.SESSION_HISTORY,
                content=f"User: {user_input}"
            ))
            self._trim_state()
            
            # 2. Run ReAct loop
            await self._execute_react_cycle()
