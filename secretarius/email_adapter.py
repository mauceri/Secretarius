from __future__ import annotations

import email
import imaplib
import os
import smtplib
import time
from dataclasses import dataclass
from email.header import decode_header, make_header
from email.message import EmailMessage
from email.utils import getaddresses, parseaddr
from typing import Any

from .channel_adapters import ChannelEvent, handle_channel_event


def _parse_allowed_senders(raw: str | None) -> set[str]:
    if not raw:
        return set()
    return {item.strip().lower() for item in raw.split(",") if item.strip()}


def _decode_header_value(value: str | None) -> str:
    if not value:
        return ""
    try:
        return str(make_header(decode_header(value)))
    except Exception:
        return value


def _extract_plain_text(msg: email.message.Message) -> str:
    if msg.is_multipart():
        for part in msg.walk():
            content_type = (part.get_content_type() or "").lower()
            if content_type != "text/plain":
                continue
            disp = (part.get("Content-Disposition") or "").lower()
            if "attachment" in disp:
                continue
            payload = part.get_payload(decode=True)
            charset = part.get_content_charset() or "utf-8"
            if isinstance(payload, bytes):
                return payload.decode(charset, errors="replace").strip()
    else:
        payload = msg.get_payload(decode=True)
        charset = msg.get_content_charset() or "utf-8"
        if isinstance(payload, bytes):
            return payload.decode(charset, errors="replace").strip()
    return ""


@dataclass(frozen=True)
class EmailConfig:
    imap_host: str
    imap_port: int
    imap_user: str
    imap_password: str
    imap_mailbox: str
    smtp_host: str
    smtp_port: int
    smtp_user: str
    smtp_password: str
    sender_email: str
    poll_interval_s: float
    allowed_senders: set[str]

    @staticmethod
    def from_env() -> "EmailConfig":
        imap_host = os.environ.get("EMAIL_IMAP_HOST", "").strip()
        imap_port = int(os.environ.get("EMAIL_IMAP_PORT", "993"))
        imap_user = os.environ.get("EMAIL_IMAP_USER", "").strip()
        imap_password = os.environ.get("EMAIL_IMAP_PASSWORD", "").strip()
        imap_mailbox = os.environ.get("EMAIL_IMAP_MAILBOX", "INBOX").strip() or "INBOX"

        smtp_host = os.environ.get("EMAIL_SMTP_HOST", "").strip()
        smtp_port = int(os.environ.get("EMAIL_SMTP_PORT", "587"))
        smtp_user = os.environ.get("EMAIL_SMTP_USER", "").strip()
        smtp_password = os.environ.get("EMAIL_SMTP_PASSWORD", "").strip()
        sender_email = os.environ.get("EMAIL_SENDER", smtp_user).strip()

        poll_interval_s = float(os.environ.get("EMAIL_POLL_INTERVAL_S", "10"))
        allowed_senders = _parse_allowed_senders(os.environ.get("EMAIL_ALLOWED_SENDERS"))

        missing = []
        if not imap_host:
            missing.append("EMAIL_IMAP_HOST")
        if not imap_user:
            missing.append("EMAIL_IMAP_USER")
        if not imap_password:
            missing.append("EMAIL_IMAP_PASSWORD")
        if not smtp_host:
            missing.append("EMAIL_SMTP_HOST")
        if not smtp_user:
            missing.append("EMAIL_SMTP_USER")
        if not smtp_password:
            missing.append("EMAIL_SMTP_PASSWORD")
        if not sender_email:
            missing.append("EMAIL_SENDER")
        if missing:
            raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

        return EmailConfig(
            imap_host=imap_host,
            imap_port=imap_port,
            imap_user=imap_user,
            imap_password=imap_password,
            imap_mailbox=imap_mailbox,
            smtp_host=smtp_host,
            smtp_port=smtp_port,
            smtp_user=smtp_user,
            smtp_password=smtp_password,
            sender_email=sender_email,
            poll_interval_s=max(1.0, poll_interval_s),
            allowed_senders=allowed_senders,
        )


class EmailPollingAdapter:
    def __init__(self, config: EmailConfig | None = None) -> None:
        self.config = config or EmailConfig.from_env()

    def run_forever(self) -> None:
        print("Email adapter started (IMAP polling).")
        while True:
            try:
                self._poll_once()
            except KeyboardInterrupt:
                print("Email adapter stopped.")
                return
            except Exception as exc:
                print(f"[email] polling error: {exc}")
            time.sleep(self.config.poll_interval_s)

    def _poll_once(self) -> None:
        with imaplib.IMAP4_SSL(self.config.imap_host, self.config.imap_port) as imap:
            imap.login(self.config.imap_user, self.config.imap_password)
            imap.select(self.config.imap_mailbox)
            typ, data = imap.uid("search", None, "UNSEEN")
            if typ != "OK" or not data or not isinstance(data[0], bytes):
                return
            uids = [uid for uid in data[0].split() if uid]
            for uid in uids:
                self._handle_uid(imap, uid)

    def _handle_uid(self, imap: imaplib.IMAP4_SSL, uid: bytes) -> None:
        typ, data = imap.uid("fetch", uid, "(RFC822)")
        if typ != "OK" or not data:
            return
        raw_msg: bytes | None = None
        for item in data:
            if isinstance(item, tuple) and len(item) >= 2 and isinstance(item[1], bytes):
                raw_msg = item[1]
                break
        if raw_msg is None:
            return

        msg = email.message_from_bytes(raw_msg)
        subject = _decode_header_value(msg.get("Subject"))
        sender_name, sender_email = parseaddr(msg.get("From", ""))
        sender_email = (sender_email or "").strip().lower()
        if not sender_email:
            imap.uid("store", uid, "+FLAGS", "(\\Seen)")
            return

        if self.config.allowed_senders and sender_email not in self.config.allowed_senders:
            imap.uid("store", uid, "+FLAGS", "(\\Seen)")
            return

        body = _extract_plain_text(msg)
        if not body:
            body = "(email sans contenu texte exploitable)"

        message_id = (msg.get("Message-ID") or "").strip() or f"uid-{uid.decode(errors='ignore')}"
        session_id = f"email:{message_id}"
        event = ChannelEvent(
            channel="email",
            user_id=sender_email,
            session_id=session_id,
            text=body.strip(),
            metadata={"subject": subject, "sender_name": sender_name},
        )

        try:
            result = handle_channel_event(event)
            reply_text = str(result.get("output_text", "")).strip() or "(aucune reponse)"
        except Exception as exc:
            reply_text = f"Erreur agent: {exc}"

        self._send_reply(
            to_email=sender_email,
            original_subject=subject,
            reply_text=reply_text,
            message_id=message_id,
            references=msg.get("References"),
        )
        imap.uid("store", uid, "+FLAGS", "(\\Seen)")

    def _send_reply(
        self,
        *,
        to_email: str,
        original_subject: str,
        reply_text: str,
        message_id: str,
        references: str | None,
    ) -> None:
        reply_subject = original_subject if original_subject.lower().startswith("re:") else f"Re: {original_subject}"
        out = EmailMessage()
        out["From"] = self.config.sender_email
        out["To"] = to_email
        out["Subject"] = reply_subject
        if message_id:
            out["In-Reply-To"] = message_id
            ref_value = (references or "").strip()
            out["References"] = f"{ref_value} {message_id}".strip()
        out.set_content(reply_text)

        with smtplib.SMTP(self.config.smtp_host, self.config.smtp_port, timeout=30) as smtp:
            smtp.starttls()
            smtp.login(self.config.smtp_user, self.config.smtp_password)
            smtp.send_message(out)

