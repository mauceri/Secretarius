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
};

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

// Diagnostic temporaire (essai 2026-11-18) : ce que répond le serveur à l'URL
// du pointeur, et les noms (pas les valeurs) des paramètres du fragment.
async function diagnostiquer(url: unknown, att: any, log: (l: string) => void): Promise<void> {
  if (typeof url !== "string" || !url) return;
  try {
    const u = new URL(url);
    const cles = u.hash.slice(1).split("&").map((x) => x.split("=")[0]).filter(Boolean).join(",");
    const essais = [u.origin + u.pathname, "https://" + u.host + u.pathname, `http://filev2.getsession.org/file/${att.id}`];
    for (const e of essais) {
      try {
        const r = await fetch(e, { redirect: "manual" });
        const b = await r.arrayBuffer();
        log(`diag ${formeUrl(e)} → HTTP ${r.status}, ${b.byteLength} octets, location ${formeUrl(r.headers.get("location"))}`);
      } catch (x) { log(`diag ${formeUrl(e)} → ${x instanceof Error ? x.message : String(x)}`); }
    }
    log(`diag fragment : clés [${cles}] ; clé ${att._key ? att._key.length : 0} o, digest ${att._digest ? att._digest.length : 0} o, taille annoncée ${att.size ?? "?"}`);
  } catch { /* diagnostic seulement */ }
}

export async function handleSessionMessage(msg: any, deps: InboundDeps): Promise<"dropped" | "duplicate" | "queued"> {
  const { cfg, store, session, log } = deps;
  if (msg.type !== "private" || !cfg.allowFrom.includes(msg.from)) {
    log(`ignoré : message de ${court(String(msg.from))} (${msg.type === "private" ? "hors liste blanche" : "groupe"})`);
    return "dropped";
  }
  if (store.seen(msg.id)) return "duplicate";

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
      await diagnostiquer(pointeurs[i]?.url, att, log);
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
