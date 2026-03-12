import asyncio
import unittest

from session_messenger_api import SessionMessengerRequest, create_session_messenger_app
from starlette.requests import Request


class FakeGateway:
    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    async def submit(self, channel: str, user_input: str) -> str:
        self.calls.append((channel, user_input))
        return "reponse-session"


class TestSessionMessengerAPI(unittest.TestCase):
    def test_message_routes_to_session_messenger_channel(self):
        gateway = FakeGateway()
        app = create_session_messenger_app(gateway=gateway, request_timeout_s=30.0)
        route = next(route for route in app.routes if getattr(route, "path", None) == "/session/message")
        request_model = SessionMessengerRequest(
            message_id="msg-1",
            sender_id="session-user-1",
            text="Bonjour depuis Session",
            bot_session_id="session-bot-123",
        )
        http_request = Request({"type": "http", "client": ("127.0.0.1", 12345), "headers": []})

        payload = asyncio.run(route.endpoint(request_model, http_request))
        self.assertEqual(payload["reply_text"], "reponse-session")
        self.assertEqual(gateway.calls, [("session_messenger", "Bonjour depuis Session")])


if __name__ == "__main__":
    unittest.main()
