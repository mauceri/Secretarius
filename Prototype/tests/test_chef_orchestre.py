import unittest
from typing import Any

from core.chef_orchestre import ChefDOrchestre
from core.models import Message, Role
from core.ports import InputGatewayInterface, LLMInterface, ToolClientInterface


class FakeLLM(LLMInterface):
    def __init__(self, responses: list[str]):
        self._responses = responses[:]

    async def generate_response(self, messages: list[Message], system_prompt: str) -> str:
        if not self._responses:
            raise RuntimeError("No fake responses left")
        return self._responses.pop(0)


class FakeToolClient(ToolClientInterface):
    def __init__(self):
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def list_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "ask_oracle",
                "description": "Ask a yes/no question",
                "inputSchema": {"type": "object"},
            }
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        self.calls.append((name, arguments))
        return "oracle: OUI"

    async def connect(self):
        return None

    async def disconnect(self):
        return None


class FakeGateway(InputGatewayInterface):
    def __init__(self):
        self.callback = None
        self.thoughts: list[str] = []
        self.messages: list[tuple[str, str]] = []

    async def display_thought(self, thought: str):
        self.thoughts.append(thought)

    async def display_message(self, role: str, content: str):
        self.messages.append((role, content))

    def set_callback(self, callback):
        self.callback = callback

    async def run(self):
        return None


class TestChefDOrchestre(unittest.IsolatedAsyncioTestCase):
    async def test_accepts_markdown_json_and_displays_final_answer(self):
        llm = FakeLLM(
            [
                """```json
{"thought":"ok","action":null,"action_input":{},"final_answer":"Bonjour"}
```"""
            ]
        )
        gateway = FakeGateway()
        orchestrator = ChefDOrchestre(llm=llm, tool_client=FakeToolClient(), gateway=gateway)

        await orchestrator.handle_user_input("Salut")

        self.assertEqual(gateway.messages[-1], ("Secretarius", "Bonjour"))

    async def test_rejects_action_and_final_answer_together_then_recovers(self):
        llm = FakeLLM(
            [
                '{"thought":"bad","action":"ask_oracle","action_input":{"question":"x"},"final_answer":"oops"}',
                '{"thought":"use tool","action":"ask_oracle","action_input":{"question":"x"},"final_answer":null}',
                '{"thought":"done","action":null,"action_input":{},"final_answer":"Termine"}',
            ]
        )
        tools = FakeToolClient()
        gateway = FakeGateway()
        orchestrator = ChefDOrchestre(llm=llm, tool_client=tools, gateway=gateway)

        await orchestrator.handle_user_input("Test")

        self.assertTrue(any("both action and final_answer" in t for t in gateway.thoughts))
        self.assertEqual(len(tools.calls), 1)
        self.assertEqual(gateway.messages[-1], ("Secretarius", "Termine"))

    async def test_state_is_trimmed(self):
        llm = FakeLLM(['{"thought":"ok","action":null,"action_input":{},"final_answer":"done"}'])
        gateway = FakeGateway()
        orchestrator = ChefDOrchestre(llm=llm, tool_client=FakeToolClient(), gateway=gateway)

        orchestrator.state.messages = [Message(role=Role.USER, content=str(i)) for i in range(130)]
        await orchestrator.handle_user_input("new")

        self.assertLessEqual(len(orchestrator.state.messages), 100)
        self.assertLessEqual(len(orchestrator.state.context_items), 200)

