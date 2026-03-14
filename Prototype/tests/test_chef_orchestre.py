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
            {
                "name": "search_text",
                "description": "Search text",
                "inputSchema": {"type": "object"},
            },
            {
                "name": "update_text",
                "description": "Update text",
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
                '"summary":{"expressions_count":2,"collection_name":"secretarius_semantic_graph","inserted_count":0,'
                '"query_count":2,"hit_lists":2},'
                '"warning":"milvus connection failed"}'
            )
        if name == "search_text":
            return '{"tool":"search_text","documents":[],"summary":{"query_count":1,"hit_lists":0,"collection_name":"secretarius_semantic_graph"}}'
        if name == "update_text":
            return (
                '{"status":"ok","tool":"update_text",'
                '"summary":{"expressions_count":2,"collection_name":"secretarius_semantic_graph","deleted_count":1,'
                '"inserted_count":2,"query_count":2,"hit_lists":2},'
                '"warning":null}'
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

    async def test_recovers_from_missing_final_brace_in_router_json(self):
        llm = FakeLLM(
            [
                '{"action":"extract_expressions","action_input":{"text":"x"}',
            ]
        )
        tools = FakeToolClient()
        gateway = FakeGateway()
        orchestrator = ChefDOrchestre(llm=llm, tool_client=tools, gateway=gateway)

        await orchestrator.handle_user_input("Test")

        self.assertEqual(len(tools.calls), 1)
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
        self.assertEqual(len(tools.calls), 1)
        self.assertEqual(tools.calls[0], ("index_text", {"text": "Bonjour"}))
        self.assertTrue(any("defaulting to index_text" in t for t in gateway.thoughts))

    async def test_direct_exp_command_bypasses_llm(self):
        llm = FakeLLM(['{"action":null,"action_input":{}}'])
        tools = FakeToolClient()
        gateway = FakeGateway()
        orchestrator = ChefDOrchestre(llm=llm, tool_client=tools, gateway=gateway)

        await orchestrator.handle_user_input("/exp Bonjour")

        self.assertEqual(llm.calls, 0)
        self.assertEqual(tools.calls, [("extract_expressions", {"text": "Bonjour"})])
        self.assertEqual(gateway.messages[-1], ("Secretarius", '{"expressions":["a","b"]}'))

    async def test_direct_index_command_bypasses_llm(self):
        llm = FakeLLM(['{"action":null,"action_input":{}}'])
        tools = FakeToolClient()
        gateway = FakeGateway()
        orchestrator = ChefDOrchestre(llm=llm, tool_client=tools, gateway=gateway)

        await orchestrator.handle_user_input("/index Corps documentaire")

        self.assertEqual(llm.calls, 0)
        self.assertEqual(tools.calls, [("index_text", {"text": "Corps documentaire"})])

    async def test_direct_index_command_accepts_newline_payload(self):
        llm = FakeLLM(['{"action":null,"action_input":{}}'])
        tools = FakeToolClient()
        gateway = FakeGateway()
        orchestrator = ChefDOrchestre(llm=llm, tool_client=tools, gateway=gateway)

        await orchestrator.handle_user_input("/index\n12/03/2026\nCorps documentaire")

        self.assertEqual(llm.calls, 0)
        self.assertEqual(tools.calls, [("index_text", {"text": "12/03/2026\nCorps documentaire"})])

    async def test_direct_req_command_bypasses_llm(self):
        llm = FakeLLM(['{"action":null,"action_input":{}}'])
        tools = FakeToolClient()
        gateway = FakeGateway()
        orchestrator = ChefDOrchestre(llm=llm, tool_client=tools, gateway=gateway)

        await orchestrator.handle_user_input("/req trou de verdure")

        self.assertEqual(llm.calls, 0)
        self.assertEqual(tools.calls, [("search_text", {"query": "trou de verdure"})])

    async def test_direct_update_command_bypasses_llm(self):
        llm = FakeLLM(['{"action":null,"action_input":{}}'])
        tools = FakeToolClient()
        gateway = FakeGateway()
        orchestrator = ChefDOrchestre(llm=llm, tool_client=tools, gateway=gateway)

        await orchestrator.handle_user_input("/update\ndoc_id: doc:test-1\nTitre\nCorps documentaire")

        self.assertEqual(llm.calls, 0)
        self.assertEqual(
            tools.calls,
            [("update_text", {"text": "doc_id: doc:test-1\nTitre\nCorps documentaire"})],
        )

    async def test_direct_command_requires_payload(self):
        llm = FakeLLM(['{"action":null,"action_input":{}}'])
        tools = FakeToolClient()
        gateway = FakeGateway()
        orchestrator = ChefDOrchestre(llm=llm, tool_client=tools, gateway=gateway)

        await orchestrator.handle_user_input("/exp")

        self.assertEqual(llm.calls, 0)
        self.assertEqual(len(tools.calls), 0)
        self.assertTrue(gateway.messages[-1][1].startswith("Commande extract_expressions sans contenu"))

    async def test_extract_expressions_drops_unsupported_args_without_rewriting(self):
        llm = FakeLLM(['{"action":"extract_expressions","action_input":{"document":"text","text":"Texte brut"}}'])
        tools = FakeToolClient()
        gateway = FakeGateway()
        orchestrator = ChefDOrchestre(llm=llm, tool_client=tools, gateway=gateway)

        await orchestrator.handle_user_input("Extraire les expressions de ce texte: Dans la plaine les baladins.")

        self.assertEqual(len(tools.calls), 1)
        tool_name, tool_args = tools.calls[0]
        self.assertEqual(tool_name, "extract_expressions")
        self.assertEqual(tool_args.get("text"), "Texte brut")
        self.assertNotIn("document", tool_args)
        self.assertTrue(any("Dropped unsupported tool args" in t for t in gateway.thoughts))

    async def test_extract_expressions_keeps_router_payload_verbatim(self):
        llm = FakeLLM(
            ['{"action":"extract_expressions","action_input":{"text":"Extraire les expressions caractéristiques de : Dans la plaine les baladins"}}']
        )
        tools = FakeToolClient()
        gateway = FakeGateway()
        orchestrator = ChefDOrchestre(llm=llm, tool_client=tools, gateway=gateway)

        await orchestrator.handle_user_input("ignored")

        self.assertEqual(len(tools.calls), 1)
        _, tool_args = tools.calls[0]
        self.assertEqual(
            tool_args.get("text"),
            "Extraire les expressions caractéristiques de : Dans la plaine les baladins",
        )

    async def test_index_text_drops_unsupported_args_without_rewriting(self):
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
        self.assertNotIn("text", tool_args)
        self.assertNotIn("collection_name", tool_args)
        self.assertNotIn("document", tool_args)
        self.assertTrue(any("Index summary: expressions=2, inserted=0" in t for t in gateway.thoughts))
        self.assertTrue(any("Dropped unsupported tool args" in t for t in gateway.thoughts))

    async def test_index_text_keeps_router_payload_verbatim_except_whitelist(self):
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
        self.assertEqual(tool_args.get("text"), "Quand je considere ces tetes entassees")
        self.assertNotIn("collection_name", tool_args)
