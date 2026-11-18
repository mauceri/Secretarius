import { describe, it, expect, beforeEach } from "bun:test";
import { mkdtempSync, readdirSync, statSync, existsSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { handleSessionMessage, type SessionLike, type InboundEvent } from "../src/inbound";
import { Store } from "../src/store";
import type { BrokerConfig } from "../src/config";

const MOI = "05" + "a".repeat(64);
const INCONNU = "05" + "b".repeat(64);

function fauxSession() {
  const calls = { read: [] as any[], getFile: 0 };
  const s: SessionLike = {
    async markMessagesAsRead(p) { calls.read.push(p); },
    async getFile(att: any) { calls.getFile++; return new File([att._octets ?? "abc"], att.name ?? "x", { type: "image/png" }); },
    async sendMessage() { return { messageHash: "h", timestamp: 1 }; },
    async showTypingIndicator() {},
    async hideTypingIndicator() {},
  };
  return { s, calls };
}

let cfg: BrokerConfig, store: Store, publies: InboundEvent[], journal: string[];
beforeEach(() => {
  const d = mkdtempSync(join(tmpdir(), "sb-in-"));
  cfg = { port: 0, allowFrom: [MOI], stateDir: d, token: "t", maxAttachmentBytes: 10 };
  store = new Store(join(d, "broker.sqlite"));
  publies = []; journal = [];
});
const deps = (s: SessionLike) => ({ session: s, store, cfg, publish: (e: InboundEvent) => publies.push(e), log: (l: string) => journal.push(l) });
const msg = (o: any) => ({ type: "private", id: "m1", from: MOI, timestamp: 1000, text: "bonjour secret", attachments: [], ...o });

describe("handleSessionMessage", () => {
  it("expéditeur hors liste : dropped, rien publié, pas d'accusé, pas de fichier, journal sans texte", async () => {
    const { s, calls } = fauxSession();
    const r = await handleSessionMessage(msg({ from: INCONNU, attachments: [{ id: "a", name: "p.png", metadata: {}, size: 3 }] }), deps(s));
    expect(r).toBe("dropped");
    expect(publies).toEqual([]);
    expect(calls.read).toEqual([]);
    expect(calls.getFile).toBe(0);
    expect(existsSync(join(cfg.stateDir, "media"))).toBe(false);
    expect(journal.join("\n")).not.toContain("bonjour secret");
    expect(journal.join("\n")).not.toContain(INCONNU);
    expect(journal.join("\n")).toContain(INCONNU.slice(0, 8));
  });
  it("même id deux fois : queued puis duplicate", async () => {
    const { s } = fauxSession();
    expect(await handleSessionMessage(msg({}), deps(s))).toBe("queued");
    expect(await handleSessionMessage(msg({}), deps(s))).toBe("duplicate");
    expect(publies.length).toBe(1);
  });
  it("pièce jointe autorisée : fichier 0600 dans media 0700, accusé une fois", async () => {
    const { s, calls } = fauxSession();
    const r = await handleSessionMessage(msg({ attachments: [{ id: "a", name: "p.png", metadata: { contentType: "image/png" }, size: 3 }] }), deps(s));
    expect(r).toBe("queued");
    const att = publies[0].attachments[0];
    expect(att.path.startsWith(join(cfg.stateDir, "media"))).toBe(true);
    expect(statSync(att.path).mode & 0o777).toBe(0o600);
    expect(statSync(join(cfg.stateDir, "media")).mode & 0o777).toBe(0o700);
    expect(att.name).toBe("p.png");
    expect(att.size).toBe(3);
    expect(calls.read).toEqual([{ from: MOI, messagesTimestamps: [1000] }]);
  });
  it("texte vide + pièce jointe : queued, text vide", async () => {
    const { s } = fauxSession();
    await handleSessionMessage(msg({ text: undefined, attachments: [{ id: "a", name: "p.png", metadata: {}, size: 3 }] }), deps(s));
    expect(publies[0].text).toBe("");
  });
  it("pièce trop grande : aucun fichier écrit, error trop-grande", async () => {
    const { s } = fauxSession();
    await handleSessionMessage(msg({ attachments: [{ id: "a", name: "gros.bin", metadata: {}, size: 11, _octets: "x".repeat(11) }] }), deps(s));
    const att = publies[0].attachments[0];
    expect(att.error).toBe("trop-grande");
    expect(att.path).toBe("");
    const media = join(cfg.stateDir, "media");
    expect(existsSync(media) ? readdirSync(media) : []).toEqual([]);
  });
  it("téléchargement impossible : queued, texte gardé, error echec-telechargement, forme de l'URL journalisée", async () => {
    const { s } = fauxSession();
    s.getFile = async () => { throw new Error("Couldn't download file from filev2 server"); };
    const m = msg({ attachments: [{ id: "123", name: "p.jpg", metadata: {}, size: 3 }],
      getContent: () => ({ dataMessage: { attachments: [{ url: "http://files.example.org/file/987654?p=abc" }] } }) });
    expect(await handleSessionMessage(m, deps(s))).toBe("queued");
    expect(publies[0].text).toBe("bonjour secret");
    expect(publies[0].attachments[0]).toMatchObject({ path: "", name: "p.jpg", error: "echec-telechargement" });
    const j = journal.join("\n");
    expect(j).toContain("files.example.org/file/N?p");
    expect(j).not.toContain("987654");
  });
  it("échec d'écriture après réception : le message n'est pas marqué vu, il sera retraité", async () => {
    const { s } = fauxSession();
    let k = 0;
    const d = { ...deps(s), ecrireFichier: (path: string, data: Uint8Array) => {
      k++;
      if (k === 1) throw new Error("ENOSPC");
      writeFileSync(path, data, { mode: 0o600 });
    } };
    const m = msg({ attachments: [{ id: "a", name: "p.png", metadata: {}, size: 3 }] });
    await expect(handleSessionMessage(m, d)).rejects.toThrow("ENOSPC");
    expect(publies).toEqual([]);
    expect(await handleSessionMessage(m, d)).toBe("queued");
    expect(publies.length).toBe(1);
  });
  it("message de groupe : dropped", async () => {
    const { s } = fauxSession();
    expect(await handleSessionMessage(msg({ type: "group", groupId: "g" }), deps(s))).toBe("dropped");
  });
  it("l'événement est mis en file non acquittée", async () => {
    const { s } = fauxSession();
    await handleSessionMessage(msg({}), deps(s));
    expect(store.pending().map((e) => e.id)).toEqual(["m1"]);
    store.ack("m1");
    expect(store.pending()).toEqual([]);
  });
});
