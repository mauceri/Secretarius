import { chmodSync, mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { randomUUID } from "node:crypto";
import type { BrokerConfig } from "./config";
import type { Store } from "./store";

export type InboundAttachment = { path: string; name: string; contentType: string; size: number; error?: "trop-grande" | "echec-telechargement" };
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
  ecrireFichier?: (path: string, data: Uint8Array) => void;
};

// Messages en cours de traitement (évite un doublon concurrent avant la mise en file).
const enCours = new Set<string>();

export const court = (id: string) => id.slice(0, 8);

// Forme de l'adresse d'une pièce jointe pour le diagnostic : hôte et chemin, chiffres masqués.
export function formeUrl(url: unknown): string {
  if (typeof url !== "string" || !url) return "(pas d'url)";
  try {
    const u = new URL(url);
    const params = [...u.searchParams.keys()].join("&");
    return `${u.host}${u.pathname.replace(/\d+/g, "N")}${params ? "?" + params : ""}${u.hash ? "#…" : ""}`;
  } catch { return "(url illisible)"; }
}

export async function handleSessionMessage(msg: any, deps: InboundDeps): Promise<"dropped" | "duplicate" | "queued"> {
  const { cfg, store, session, log } = deps;
  if (msg.type !== "private" || !cfg.allowFrom.includes(msg.from)) {
    log(`ignoré : message de ${court(String(msg.from))} (${msg.type === "private" ? "hors liste blanche" : "groupe"})`);
    return "dropped";
  }
  if (store.seen(msg.id) || enCours.has(msg.id)) return "duplicate";
  enCours.add(msg.id);
  try {
    return await traiter(msg, deps);
  } finally {
    enCours.delete(msg.id);
  }
}

async function traiter(msg: any, deps: InboundDeps): Promise<"queued"> {
  const { cfg, store, session, log } = deps;

  const attachments: InboundAttachment[] = [];
  const pointeurs: any[] = (() => { try { return msg.getContent?.()?.dataMessage?.attachments ?? []; } catch { return []; } })();
  for (const [i, att] of (msg.attachments ?? []).entries()) {
    const name = String(att.name ?? "piece-jointe");
    const contentType = String(att.metadata?.contentType ?? "application/octet-stream");
    if ((att.size ?? 0) > cfg.maxAttachmentBytes) {
      attachments.push({ path: "", name, contentType, size: att.size, error: "trop-grande" });
      continue;
    }
    let file: File;
    try {
      file = await session.getFile(att);
    } catch (e) {
      log(`pièce jointe non téléchargée (${e instanceof Error ? e.message : String(e)}) ; id ${String(att.id).length} chiffres, url ${formeUrl(pointeurs[i]?.url)}`);
      attachments.push({ path: "", name, contentType, size: att.size ?? 0, error: "echec-telechargement" });
      continue;
    }
    if (file.size > cfg.maxAttachmentBytes) {
      attachments.push({ path: "", name, contentType, size: file.size, error: "trop-grande" });
      continue;
    }
    const media = join(cfg.stateDir, "media");
    mkdirSync(media, { recursive: true, mode: 0o700 });
    chmodSync(media, 0o700);
    const path = join(media, randomUUID());
    (deps.ecrireFichier ?? ((p: string, d: Uint8Array) => writeFileSync(p, d, { mode: 0o600 })))(path, new Uint8Array(await file.arrayBuffer()));
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
