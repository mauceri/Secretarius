import { chmodSync, mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { randomUUID } from "node:crypto";
import type { BrokerConfig } from "./config";
import type { Store } from "./store";

export type InboundAttachment = { path: string; name: string; contentType: string; size: number; error?: "trop-grande" };
export type InboundEvent = { id: string; from: string; timestamp: number; text: string; attachments: InboundAttachment[] };

// Sous-ensemble de Session (@session.js/client 0.0.57) utilisé par le broker.
export interface SessionLike {
  markMessagesAsRead(p: { from: string; messagesTimestamps: number[] }): Promise<void>;
  getFile(att: any): Promise<File>;
  sendMessage(p: { to: string; text: string }): Promise<{ messageHash: string; timestamp: number }>;
  showTypingIndicator(p: { conversation: string }): Promise<void>;
  hideTypingIndicator(p: { conversation: string }): Promise<void>;
}

export type InboundDeps = {
  session: SessionLike;
  store: Store;
  cfg: BrokerConfig;
  publish(evt: InboundEvent): void;
  log(line: string): void;
};

export const court = (id: string) => id.slice(0, 8);

export async function handleSessionMessage(msg: any, deps: InboundDeps): Promise<"dropped" | "duplicate" | "queued"> {
  const { cfg, store, session, log } = deps;
  if (msg.type !== "private" || !cfg.allowFrom.includes(msg.from)) {
    log(`ignoré : message de ${court(String(msg.from))} (${msg.type === "private" ? "hors liste blanche" : "groupe"})`);
    return "dropped";
  }
  if (store.seen(msg.id)) return "duplicate";

  const attachments: InboundAttachment[] = [];
  for (const att of msg.attachments ?? []) {
    const name = String(att.name ?? "piece-jointe");
    const contentType = String(att.metadata?.contentType ?? "application/octet-stream");
    if ((att.size ?? 0) > cfg.maxAttachmentBytes) {
      attachments.push({ path: "", name, contentType, size: att.size, error: "trop-grande" });
      continue;
    }
    const file = await session.getFile(att);
    if (file.size > cfg.maxAttachmentBytes) {
      attachments.push({ path: "", name, contentType, size: file.size, error: "trop-grande" });
      continue;
    }
    const media = join(cfg.stateDir, "media");
    mkdirSync(media, { recursive: true, mode: 0o700 });
    chmodSync(media, 0o700);
    const path = join(media, randomUUID());
    writeFileSync(path, new Uint8Array(await file.arrayBuffer()), { mode: 0o600 });
    attachments.push({ path, name, contentType: file.type || contentType, size: file.size });
  }

  const evt: InboundEvent = { id: msg.id, from: msg.from, timestamp: msg.timestamp, text: msg.text ?? "", attachments };
  store.enqueue(evt);
  try {
    await session.markMessagesAsRead({ from: msg.from, messagesTimestamps: [msg.timestamp] });
  } catch {
    log(`accusé de lecture non envoyé à ${court(msg.from)}`);
  }
  log(`reçu de ${court(msg.from)} : ${evt.text.length} car., ${attachments.length} pièce(s)`);
  deps.publish(evt);
  return "queued";
}
