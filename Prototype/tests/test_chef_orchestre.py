import unittest
from typing import Any

from core.chef_orchestre import ChefDOrchestre
from core.models import Message, Role
from core.ports import InputGatewayInterface, LLMInterface, ToolClientInterface


class FakeLLM(LLMInterface):
    def __init__(self, responses: list[str]):
        self._responses = responses[:]
        self.calls = 0

    async def generate_response(self, messages: list[Message], system_prompt: str) -> str:
        self.calls += 1
        if not self._responses:
            raise RuntimeError("No fake responses left")
        return self._responses.pop(0)


class FakeToolClient(ToolClientInterface):
    def __init__(self):
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def list_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "extract_expressions",
                "description": "Extract expressions",
                "inputSchema": {"type": "object"},
            },
            {
                "name": "ask_oracle",
                "description": "Ask a yes/no question",
                "inputSchema": {"type": "object"},
            },
            {
                "name": "index_text",
                "description": "Index text",
                "inputSchema": {"type": "object"},
            },
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        self.calls.append((name, arguments))
        if name == "extract_expressions":
            return '{"expressions":["a","b"]}'
        if name == "ask_oracle":
            return "oracle: OUI"
        if name == "index_text":
            return (
                '{"status":"ok","tool":"index_text",'
                '"extract":{"expressions":["a","b"]},'
                '"index":{"collection_name":"secretarius_semantic_graph","inserted_count":0,'
                '"warning":"milvus connection failed"}}'
            )
        return "unknown"

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
    async def test_accepts_markdown_json_and_returns_tool_output_directly(self):
        llm = FakeLLM(
            [
                """```json
{"action":"extract_expressions","action_input":{"text":"Bonjour"}}
```"""
            ]
        )
        tools = FakeToolClient()
        gateway = FakeGateway()
        orchestrator = ChefDOrchestre(llm=llm, tool_client=tools, gateway=gateway)

        await orchestrator.handle_user_input("Salut")

        self.assertEqual(llm.calls, 1)
        self.assertEqual(len(tools.calls), 1)
        self.assertEqual(gateway.messages[-1], ("Secretarius", '{"expressions":["a","b"]}'))

    async def test_ignores_final_answer_and_executes_action(self):
        llm = FakeLLM(
            [
                '{"action":"extract_expressions","action_input":{"text":"x"},"final_answer":"oops"}',
            ]
        )
        tools = FakeToolClient()
        gateway = FakeGateway()
        orchestrator = ChefDOrchestre(llm=llm, tool_client=tools, gateway=gateway)

        await orchestrator.handle_user_input("Test")

        self.assertEqual(len(tools.calls), 1)
        self.assertTrue(any("Ignored model final_answer" in t for t in gateway.thoughts))
        self.assertEqual(gateway.messages[-1], ("Secretarius", '{"expressions":["a","b"]}'))

    async def test_state_is_trimmed(self):
        llm = FakeLLM(['{"action":"extract_expressions","action_input":{"text":"done"}}'])
        gateway = FakeGateway()
        orchestrator = ChefDOrchestre(llm=llm, tool_client=FakeToolClient(), gateway=gateway)

        orchestrator.state.messages = [Message(role=Role.USER, content=str(i)) for i in range(130)]
        await orchestrator.handle_user_input("new")

        self.assertLessEqual(len(orchestrator.state.messages), 100)
        self.assertLessEqual(len(orchestrator.state.context_items), 200)

    async def test_blocks_ask_oracle_when_not_explicitly_requested(self):
        llm = FakeLLM(
            [
                '{"action":"ask_oracle","action_input":{"question":"Fera-t-il beau ?"}}',
            ]
        )
        tools = FakeToolClient()
        gateway = FakeGateway()
        orchestrator = ChefDOrchestre(llm=llm, tool_client=tools, gateway=gateway)

        await orchestrator.handle_user_input("Fera-t-il beau aujourd'hui ?")

        self.assertEqual(len(tools.calls), 0)
        self.assertTrue(any("Blocked ask_oracle" in t for t in gateway.thoughts))
        self.assertEqual(
            gateway.messages[-1],
            ("Secretarius", "Aucun outil ne correspond a votre demande."),
        )

    async def test_returns_fallback_when_no_tool_matches(self):
        llm = FakeLLM(['{"action":null,"action_input":{}}'])
        tools = FakeToolClient()
        gateway = FakeGateway()
        orchestrator = ChefDOrchestre(llm=llm, tool_client=tools, gateway=gateway)

        await orchestrator.handle_user_input("Bonjour")

        self.assertEqual(llm.calls, 1)
        self.assertEqual(len(tools.calls), 0)
        self.assertEqual(gateway.messages[-1], ("Secretarius", "Aucun outil ne correspond a votre demande."))

    async def test_extract_expressions_recovers_when_text_missing(self):
        llm = FakeLLM(['{"action":"extract_expressions","action_input":{"document":"text"}}'])
        tools = FakeToolClient()
        gateway = FakeGateway()
        orchestrator = ChefDOrchestre(llm=llm, tool_client=tools, gateway=gateway)

        user_text = "Extraire les expressions de ce texte: Dans la plaine les baladins."
        await orchestrator.handle_user_input(user_text)

        self.assertEqual(len(tools.calls), 1)
        tool_name, tool_args = tools.calls[0]
        self.assertEqual(tool_name, "extract_expressions")
        self.assertEqual(tool_args.get("text"), "Dans la plaine les baladins.")

    async def test_extract_expressions_strips_instruction_prefix(self):
        llm = FakeLLM(
            ['{"action":"extract_expressions","action_input":{"text":"Extraire les expressions caractéristiques de : Dans la plaine les baladins"}}']
        )
        tools = FakeToolClient()
        gateway = FakeGateway()
        orchestrator = ChefDOrchestre(llm=llm, tool_client=tools, gateway=gateway)

        await orchestrator.handle_user_input("ignored")

        self.assertEqual(len(tools.calls), 1)
        _, tool_args = tools.calls[0]
        self.assertEqual(tool_args.get("text"), "Dans la plaine les baladins")

    async def test_index_text_recovers_from_document_string_and_drops_bad_collection_name(self):
        llm = FakeLLM(
            [
                '{"action":"index_text","action_input":{"collection_name":"text","document":"Indexer le texte : Dans la plaine les baladins"}}',
            ]
        )
        tools = FakeToolClient()
        gateway = FakeGateway()
        orchestrator = ChefDOrchestre(llm=llm, tool_client=tools, gateway=gateway)

        await orchestrator.handle_user_input("Indexer le texte : Dans la plaine les baladins")

        self.assertEqual(len(tools.calls), 1)
        tool_name, tool_args = tools.calls[0]
        self.assertEqual(tool_name, "index_text")
        self.assertEqual(tool_args.get("text"), "Dans la plaine les baladins")
        self.assertNotIn("collection_name", tool_args)
        self.assertNotIn("document", tool_args)
        self.assertTrue(any("Index summary:" in t for t in gateway.thoughts))

    async def test_index_text_ignores_collection_name_unless_user_requests_it(self):
        llm = FakeLLM(
            [
                '{"action":"index_text","action_input":{"collection_name":"tetes_charniers","text":"Quand je considere ces tetes entassees"}}',
            ]
        )
        tools = FakeToolClient()
        gateway = FakeGateway()
        orchestrator = ChefDOrchestre(llm=llm, tool_client=tools, gateway=gateway)

        await orchestrator.handle_user_input("Indexer : Quand je considere ces tetes entassees")

        self.assertEqual(len(tools.calls), 1)
        _, tool_args = tools.calls[0]
        self.assertNotIn("collection_name", tool_args)

    async def test_index_text_prefers_user_multiline_text_over_llm_rewrite(self):
        llm = FakeLLM(
            [
                '{"action":"index_text","action_input":{"text":"Indexer le texte : version alteree sur une ligne"}}',
            ]
        )
        tools = FakeToolClient()
        gateway = FakeGateway()
        orchestrator = ChefDOrchestre(llm=llm, tool_client=tools, gateway=gateway)

        user_text = "Indexer le texte :\nVers un.\nVers deux."
        await orchestrator.handle_user_input(user_text)

        self.assertEqual(len(tools.calls), 1)
        _, tool_args = tools.calls[0]
        self.assertEqual(tool_args.get("text"), "Vers un.\nVers deux.")
