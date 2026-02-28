from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .agent_runtime import get_runtime

LOGGER = logging.getLogger("secretarius.channel")


@dataclass(frozen=True)
class ChannelEvent:
    channel: str
    user_id: str
    session_id: str
    text: str
    metadata: dict[str, Any]


def handle_channel_event(event: ChannelEvent) -> dict[str, Any]:
    LOGGER.info(
        "channel_event start channel=%s user_id=%s session_id=%s text_len=%d",
        event.channel,
        event.user_id,
        event.session_id,
        len(event.text or ""),
    )
    runtime = get_runtime()
    output = runtime.run_from_prompt(
        event.text,
        channel=event.channel,
        session_id=event.session_id,
    )
    LOGGER.info(
        "channel_event done channel=%s session_id=%s output_len=%d",
        event.channel,
        event.session_id,
        len(output),
    )
    return {
        "channel": event.channel,
        "user_id": event.user_id,
        "session_id": event.session_id,
        "output_text": output,
        "metadata": event.metadata,
    }


# Stubs de canaux futurs (connecteurs a implémenter plus tard).
CHANNEL_STUBS = {
    "telegram": "todo",
    "messenger": "todo",
    "whatsapp": "todo",
    "email": "todo",
}
