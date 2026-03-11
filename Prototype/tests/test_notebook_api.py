import asyncio
import unittest

from notebook_api import create_notebook_app
from notebook_api import ChatCompletionRequest
from starlette.requests import Request


class FakeGateway:
    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    async def submit(self, channel: str, user_input: str) -> str:
        self.calls.append((channel, user_input))
        return '{"expressions":["a","b"]}'


class TestNotebookAPI(unittest.TestCase):
    def test_chat_completion_routes_to_notebook_channel(self):
        gateway = FakeGateway()
        app = create_notebook_app(gateway=gateway, model_id="secretarius-notebook")
        route = next(route for route in app.routes if getattr(route, "path", None) == "/v1/chat/completions")
        request_model = ChatCompletionRequest(
            model="secretarius-notebook",
            messages=[{"role": "user", "content": "Expressions : bonjour"}],
            stream=False,
        )
        http_request = Request({"type": "http", "client": ("127.0.0.1", 12345), "headers": []})

        payload = asyncio.run(route.endpoint(request_model, http_request))
        self.assertEqual(
            payload["choices"][0]["message"]["content"],
            '{"expressions":["a","b"]}',
        )
        self.assertEqual(gateway.calls, [("notebook", "Expressions : bonjour")])


if __name__ == "__main__":
    unittest.main()
