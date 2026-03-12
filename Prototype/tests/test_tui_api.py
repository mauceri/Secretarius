import asyncio
import unittest

from starlette.requests import Request

from tui_api import TUIMessageRequest, create_tui_app


class FakeGateway:
    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    async def submit_with_trace(self, channel: str, user_input: str) -> dict:
        self.calls.append((channel, user_input))
        return {
            "reply_text": "reponse-tui",
            "thoughts": ["analyse 1", "analyse 2"],
            "messages": [{"role": "assistant", "content": "reponse-tui"}],
        }


class TestTUIAPI(unittest.TestCase):
    def test_message_routes_to_tui_channel_with_trace(self):
        gateway = FakeGateway()
        app = create_tui_app(gateway=gateway, request_timeout_s=30.0)
        route = next(route for route in app.routes if getattr(route, "path", None) == "/tui/message")
        request_model = TUIMessageRequest(text="Bonjour depuis TUI")
        http_request = Request({"type": "http", "client": ("127.0.0.1", 12345), "headers": []})

        payload = asyncio.run(route.endpoint(request_model, http_request))
        self.assertEqual(payload["reply_text"], "reponse-tui")
        self.assertEqual(payload["thoughts"], ["analyse 1", "analyse 2"])
        self.assertEqual(gateway.calls, [("tui", "Bonjour depuis TUI")])


if __name__ == "__main__":
    unittest.main()
