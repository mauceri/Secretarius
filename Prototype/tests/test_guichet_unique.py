import asyncio
import unittest

from adapters.input.guichet_unique import GuichetUnique


class TestGuichetUniqueDedup(unittest.IsolatedAsyncioTestCase):
    async def test_deduplicates_concurrent_same_channel_and_text(self):
        gateway = GuichetUnique()
        callback_calls = 0

        async def callback(user_input: str) -> None:
            nonlocal callback_calls
            callback_calls += 1
            await asyncio.sleep(0.05)
            await gateway.display_message("Secretarius", f"ok:{user_input}")

        gateway.set_callback(callback)

        res1, res2 = await asyncio.gather(
            gateway.submit("openwebui", "Bonjour"),
            gateway.submit("openwebui", "Bonjour"),
        )

        self.assertEqual(callback_calls, 1)
        self.assertEqual(res1, "ok:Bonjour")
        self.assertEqual(res2, "ok:Bonjour")

    async def test_allows_sequential_same_text(self):
        gateway = GuichetUnique(recent_result_ttl_s=0.0)
        callback_calls = 0

        async def callback(user_input: str) -> None:
            nonlocal callback_calls
            callback_calls += 1
            await gateway.display_message("Secretarius", f"ok:{user_input}:{callback_calls}")

        gateway.set_callback(callback)

        res1 = await gateway.submit("openwebui", "Bonjour")
        res2 = await gateway.submit("openwebui", "Bonjour")

        self.assertEqual(callback_calls, 2)
        self.assertEqual(res1, "ok:Bonjour:1")
        self.assertEqual(res2, "ok:Bonjour:2")

    async def test_returns_cached_recent_result_for_sequential_duplicate(self):
        gateway = GuichetUnique(recent_result_ttl_s=15.0)
        callback_calls = 0

        async def callback(user_input: str) -> None:
            nonlocal callback_calls
            callback_calls += 1
            await gateway.display_message("Secretarius", f"ok:{user_input}:{callback_calls}")

        gateway.set_callback(callback)

        res1 = await gateway.submit("openwebui", "Bonjour")
        res2 = await gateway.submit("openwebui", "Bonjour")

        self.assertEqual(callback_calls, 1)
        self.assertEqual(res1, "ok:Bonjour:1")
        self.assertEqual(res2, "ok:Bonjour:1")

    async def test_submit_with_trace_collects_thoughts_and_messages(self):
        gateway = GuichetUnique(recent_result_ttl_s=0.0)

        async def callback(user_input: str) -> None:
            await gateway.display_thought(f"analyse:{user_input}")
            await gateway.display_message("Secretarius", f"ok:{user_input}")

        gateway.set_callback(callback)

        payload = await gateway.submit_with_trace("tui", "Bonjour")

        self.assertEqual(payload["reply_text"], "ok:Bonjour")
        self.assertEqual(payload["thoughts"], ["analyse:Bonjour"])
        self.assertEqual(
            payload["messages"],
            [{"role": "Secretarius", "content": "ok:Bonjour"}],
        )


if __name__ == "__main__":
    unittest.main()
