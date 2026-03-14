import tempfile
import textwrap
import unittest
from pathlib import Path

import app_runtime


class _FakeGateway:
    async def display_thought(self, thought: str):
        _ = thought

    async def display_message(self, role: str, content: str):
        _ = (role, content)

    def set_callback(self, callback):
        _ = callback

    async def run(self):
        return None


class _FakeMCPClient:
    last_instance = None

    def __init__(self, *, command, args, cwd, env):
        self.command = command
        self.args = args
        self.cwd = cwd
        self.env = env
        _FakeMCPClient.last_instance = self

    async def connect(self):
        return None

    async def disconnect(self):
        return None


class _FakeLLMAdapter:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class TestAppRuntime(unittest.IsolatedAsyncioTestCase):
    async def test_build_runtime_injects_milvus_collection_from_config(self):
        config_text = textwrap.dedent(
            """
            llm:
              base_url: "http://localhost:11434"
              model: "qwen3.5:2B"

            mcp_servers:
              secretarius:
                command: "python"
                args: ["-m", "secretarius_local.mcp_server"]
                log_file: "logs/mcp_server.log"
                search_min_score: 0.42
                collection_name: "secretarius_semantic_graph_test"
            """
        ).strip()

        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.yaml"
            config_path.write_text(config_text, encoding="utf-8")

            original_client = app_runtime.StdioMCPClient
            original_llm = app_runtime.OllamaAdapter
            app_runtime.StdioMCPClient = _FakeMCPClient
            app_runtime.OllamaAdapter = _FakeLLMAdapter
            try:
                runtime = await app_runtime.build_runtime(_FakeGateway(), config_path=config_path)
            finally:
                app_runtime.StdioMCPClient = original_client
                app_runtime.OllamaAdapter = original_llm

        self.assertIn("orchestrator", runtime)
        self.assertIsNotNone(_FakeMCPClient.last_instance)
        self.assertEqual(
            _FakeMCPClient.last_instance.env.get("SECRETARIUS_MILVUS_COLLECTION"),
            "secretarius_semantic_graph_test",
        )
        self.assertEqual(
            _FakeMCPClient.last_instance.env.get("SECRETARIUS_MILVUS_MIN_SCORE"),
            "0.42",
        )


if __name__ == "__main__":
    unittest.main()
