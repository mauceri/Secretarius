import { generateSeedHex } from "@session.js/keypair";
import { encode } from "@session.js/mnemonic";
import { Poller, Session, ready } from "@session.js/client";
import path from "path";
import * as fs from "fs";

import { sequelize, SessionMessage } from "./db";

const projectRoot = path.resolve(import.meta.dir, "..");
const configFilePath = process.env.SESSION_MESSENGER_CONFIG_PATH ||
  path.join(projectRoot, "session_messenger_bridge", "session_bot_config.sh");
const journalPath = process.env.SESSION_MESSENGER_JOURNAL_PATH ||
  path.join(projectRoot, "logs", "session_messenger.log");
const apiBaseUrl = (process.env.SESSION_MESSENGER_API_URL || "http://127.0.0.1:8002").replace(/\/+$/, "");
const apiUrl = `${apiBaseUrl}/session/message`;

function appendJournalLine(stream: string, role: string, text: string) {
  const timestamp = new Date().toISOString().replace(/\.\d{3}Z$/, "Z");
  const normalized = `${text}`.replace(/\r\n/g, "\n").replace(/\r/g, "\n");
  const line = `${timestamp}\tsession_messenger\t${stream}\t${role}\t${normalized}\n`;
  fs.mkdirSync(path.dirname(journalPath), { recursive: true });
  fs.appendFileSync(journalPath, line, "utf-8");
}

function saveMnemonicToConfigFile(mnemonic: string) {
  const envVarEntry = `export SESSION_BOT_MNEMONIC="${mnemonic}"\n`;
  fs.mkdirSync(path.dirname(configFilePath), { recursive: true });
  fs.writeFileSync(configFilePath, envVarEntry, "utf-8");
}

function loadMnemonicFromConfigFile() {
  if (!fs.existsSync(configFilePath)) {
    return null;
  }
  const content = fs.readFileSync(configFilePath, "utf-8");
  const match = content.match(/export SESSION_BOT_MNEMONIC="(.+?)"/);
  return match ? match[1] : null;
}

await ready;
await sequelize.authenticate();
await sequelize.sync();

let mnemonic = process.env.SESSION_BOT_MNEMONIC || loadMnemonicFromConfigFile();
if (!mnemonic) {
  mnemonic = encode(generateSeedHex());
  saveMnemonicToConfigFile(mnemonic);
  appendJournalLine("SYSTEM", "BOT", "generated new Session mnemonic");
}

const session = new Session();
session.setMnemonic(mnemonic, process.env.SESSION_BOT_DISPLAY_NAME || "Secretarius");
session.addPoller(new Poller());

const botSessionId = session.getSessionID();
appendJournalLine("SYSTEM", "BOT", `bot_session_id=${botSessionId}`);
console.log("Session Messenger bot Session ID:", botSessionId);

session.on("message", async (message) => {
  const messageId = `${message.id || ""}`.trim();
  if (!messageId) {
    appendJournalLine("SYSTEM", "DEDUP", "ignored message without id");
    return;
  }

  const existing = await SessionMessage.findByPk(messageId);
  if (existing) {
    appendJournalLine("SYSTEM", "DEDUP", `ignored duplicate message_id=${messageId}`);
    return;
  }
  await SessionMessage.create({ messageId, timestamp: new Date() });

  const text = `${message.getContent().dataMessage?.body || ""}`.trim();
  if (!text) {
    appendJournalLine("CHAT", "USER", `ignored empty message from=${message.from}`);
    return;
  }

  appendJournalLine("CHAT", "USER", `from=${message.from} message_id=${messageId}\n${text}`);

  let replyText = "Désolé, une erreur est survenue lors du traitement de votre demande.";
  try {
    const response = await fetch(apiUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        message_id: messageId,
        sender_id: message.from,
        text,
        bot_session_id: botSessionId,
        attachments: [],
      }),
    });

    if (!response.ok) {
      const responseText = await response.text();
      appendJournalLine("SYSTEM", "ERROR", `api_status=${response.status} body=${responseText}`);
    } else {
      const payload = await response.json();
      replyText = `${payload.reply_text || ""}`.trim() || "Aucune réponse n'a été produite.";
    }
  } catch (error) {
    appendJournalLine("SYSTEM", "ERROR", `api_request_failed=${error}`);
  }

  appendJournalLine("CHAT", "ASSISTANT", `to=${message.from} message_id=${messageId}\n${replyText}`);
  await session.sendMessage({
    to: message.from,
    text: replyText,
  });
});
