import asyncio
import unittest

from memos_api import create_memos_app
from starlette.requests import Request


class FakeGateway:
    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    async def submit(self, channel: str, user_input: str) -> str:
        self.calls.append((channel, user_input))
        return "resultat-secretarius"


class FakeMemosClient:
    def __init__(self):
        self.calls: list[tuple[str, str, str]] = []

    async def create_comment(self, memo_name: str, content: str, visibility: str):
        self.calls.append((memo_name, content, visibility))
        return {"name": "memos/comment-1"}


def make_request(payload: bytes, query_string: bytes = b"", headers: list[tuple[bytes, bytes]] | None = None) -> Request:
    body = payload
    sent = False

    async def receive():
        nonlocal sent
        if sent:
            return {"type": "http.request", "body": b"", "more_body": False}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(
        {
            "type": "http",
            "client": ("127.0.0.1", 12345),
            "method": "POST",
            "path": "/memos/webhook",
            "query_string": query_string,
            "headers": headers or [(b"content-type", b"application/json")],
        },
        receive=receive,
    )


class TestMemosAPI(unittest.TestCase):
    def test_webhook_routes_supported_command_and_publishes_comment(self):
        gateway = FakeGateway()
        memos_client = FakeMemosClient()
        app = create_memos_app(
            gateway=gateway,
            memos_base_url="http://127.0.0.1:5230",
            memos_access_token="secret-token",
            memos_client=memos_client,
        )
        route = next(route for route in app.routes if getattr(route, "path", None) == "/memos/webhook")
        request = make_request(
            b'{"eventType":"memo.created","memo":{"name":"memos/note-1","content":"/req cavalerie rouge","visibility":"PRIVATE"}}'
        )

        payload = asyncio.run(route.endpoint(request))
        self.assertEqual(payload["status"], "processed")
        self.assertEqual(gateway.calls, [("memos", "/req cavalerie rouge")])
        self.assertEqual(
            memos_client.calls,
            [("memos/note-1", "Secretarius\n\nresultat-secretarius", "PRIVATE")],
        )

    def test_webhook_routes_secretarius_markdown_block(self):
        gateway = FakeGateway()
        memos_client = FakeMemosClient()
        app = create_memos_app(
            gateway=gateway,
            memos_base_url="http://127.0.0.1:5230",
            memos_access_token="secret-token",
            memos_client=memos_client,
        )
        route = next(route for route in app.routes if getattr(route, "path", None) == "/memos/webhook")
        request = make_request(
            b'{"eventType":"memo.created","memo":{"name":"memos/note-1","content":"# Cavalerie rouge\\n\\n```secretarius\\naction: index\\ndoc_id: doc:boudienny-001\\ntype_note: lecture\\ntags: URSS, cavalerie\\n```\\n\\nTexte de la note..."}}'
        )

        payload = asyncio.run(route.endpoint(request))
        self.assertEqual(payload["status"], "processed")
        self.assertEqual(
            gateway.calls,
            [
                (
                    "memos",
                    "/index\n"
                    "doc_id: doc:boudienny-001\n"
                    "type_note: lecture\n"
                    "#URSS #cavalerie\n"
                    "# Cavalerie rouge\n\nTexte de la note...",
                )
            ],
        )

    def test_webhook_rejects_invalid_secretarius_markdown(self):
        gateway = FakeGateway()
        memos_client = FakeMemosClient()
        app = create_memos_app(
            gateway=gateway,
            memos_base_url="http://127.0.0.1:5230",
            memos_access_token="secret-token",
            memos_client=memos_client,
        )
        route = next(route for route in app.routes if getattr(route, "path", None) == "/memos/webhook")
        request = make_request(
            b'{"eventType":"memo.created","memo":{"name":"memos/note-1","content":"```secretarius\\naction: update\\n```\\n\\nTexte sans doc_id"}}'
        )

        with self.assertRaises(Exception) as ctx:
            asyncio.run(route.endpoint(request))

        self.assertEqual(getattr(ctx.exception, "status_code", None), 400)

    def test_webhook_ignores_non_command_memo(self):
        gateway = FakeGateway()
        memos_client = FakeMemosClient()
        app = create_memos_app(
            gateway=gateway,
            memos_base_url="http://127.0.0.1:5230",
            memos_access_token="secret-token",
            memos_client=memos_client,
        )
        route = next(route for route in app.routes if getattr(route, "path", None) == "/memos/webhook")
        request = make_request(
            b'{"eventType":"memo.created","memo":{"name":"memos/note-1","content":"note libre sans commande"}}'
        )

        payload = asyncio.run(route.endpoint(request))
        self.assertEqual(payload["status"], "ignored")
        self.assertEqual(payload["reason"], "no_supported_command")
        self.assertEqual(gateway.calls, [])
        self.assertEqual(memos_client.calls, [])

    def test_webhook_ignores_comment_events(self):
        gateway = FakeGateway()
        memos_client = FakeMemosClient()
        app = create_memos_app(
            gateway=gateway,
            memos_base_url="http://127.0.0.1:5230",
            memos_access_token="secret-token",
            memos_client=memos_client,
        )
        route = next(route for route in app.routes if getattr(route, "path", None) == "/memos/webhook")
        request = make_request(
            b'{"eventType":"memo.created","memo":{"name":"memos/comment-1","parent":"memos/note-1","content":"/req cavalerie rouge"}}'
        )

        payload = asyncio.run(route.endpoint(request))
        self.assertEqual(payload["status"], "ignored")
        self.assertEqual(payload["reason"], "comment_event")
        self.assertEqual(gateway.calls, [])
        self.assertEqual(memos_client.calls, [])

    def test_webhook_requires_expected_query_token(self):
        gateway = FakeGateway()
        memos_client = FakeMemosClient()
        app = create_memos_app(
            gateway=gateway,
            memos_base_url="http://127.0.0.1:5230",
            memos_access_token="secret-token",
            webhook_token="expected-token",
            memos_client=memos_client,
        )
        route = next(route for route in app.routes if getattr(route, "path", None) == "/memos/webhook")
        request = make_request(
            b'{"eventType":"memo.created","memo":{"name":"memos/note-1","content":"/req cavalerie rouge"}}',
            query_string=b"token=bad-token",
        )

        with self.assertRaises(Exception) as ctx:
            asyncio.run(route.endpoint(request))

        self.assertEqual(getattr(ctx.exception, "status_code", None), 401)


if __name__ == "__main__":
    unittest.main()
