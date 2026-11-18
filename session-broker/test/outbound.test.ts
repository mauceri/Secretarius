import { describe, it, expect } from "bun:test";
import { sendText, NotAllowedError } from "../src/outbound";
import { createServer } from "../src/http";

const MOI = "05" + "a".repeat(64);
const INCONNU = "05" + "b".repeat(64);
const cfg = { port: 0, allowFrom: [MOI], stateDir: "/tmp/x", token: "t", maxAttachmentBytes: 10 };

function session(echecs: number) {
  const s: any = { appels: 0, typing: [] as string[] };
  s.sendMessage = async () => { s.appels++; if (s.appels <= echecs) throw new Error("réseau"); return { messageHash: "h1", timestamp: 42 }; };
  s.showTypingIndicator = async (p: any) => { s.typing.push("on:" + p.conversation); };
  s.hideTypingIndicator = async (p: any) => { s.typing.push("off:" + p.conversation); };
  return s;
}
const sleep = async () => {};

describe("sendText", () => {
  it("refuse un destinataire hors liste sans appeler sendMessage", async () => {
    const s = session(0);
    await expect(sendText(INCONNU, "x", { session: s, cfg, sleep, log: () => {} })).rejects.toBeInstanceOf(NotAllowedError);
    expect(s.appels).toBe(0);
  });
  it("deux échecs puis succès : 3 appels", async () => {
    const s = session(2);
    expect(await sendText(MOI, "x", { session: s, cfg, sleep, log: () => {} })).toEqual({ messageId: "h1", timestamp: 42 });
    expect(s.appels).toBe(3);
  });
  it("trois échecs : lève, journal sans le texte", async () => {
    const s = session(3); const j: string[] = [];
    await expect(sendText(MOI, "texte privé", { session: s, cfg, sleep, log: (l) => j.push(l) })).rejects.toThrow();
    expect(s.appels).toBe(3);
    expect(j.join("\n")).not.toContain("texte privé");
  });
});

describe("routes /send et /typing", () => {
  const H = { Authorization: "Bearer t", "Content-Type": "application/json" };
  const app = (s: any, status: any = { lastPollOkAt: null, lastSendOkAt: null, sessionId: "05bot" }) =>
    createServer({ cfg, status, session: s, sleep, log: () => {} } as any);
  const post = (a: any, path: string, body: unknown) => a.fetch(new Request(`http://127.0.0.1${path}`, { method: "POST", headers: H, body: JSON.stringify(body) }));

  it("/send hors liste : 403", async () => {
    const s = session(0);
    expect((await post(app(s), "/send", { to: INCONNU, text: "x" })).status).toBe(403);
    expect(s.appels).toBe(0);
  });
  it("/send réussi : 200 et lastSendOkAt mis à jour", async () => {
    const status = { lastPollOkAt: null, lastSendOkAt: null as string | null, sessionId: "05bot" };
    const r = await post(app(session(0), status), "/send", { to: MOI, text: "x" });
    expect(r.status).toBe(200);
    expect(await r.json()).toEqual({ messageId: "h1", timestamp: 42 });
    expect(status.lastSendOkAt).not.toBeNull();
  });
  it("/send après trois échecs : 502", async () => {
    expect((await post(app(session(3)), "/send", { to: MOI, text: "x" })).status).toBe(502);
  });
  it("/typing : 204, et 403 hors liste", async () => {
    const s = session(0);
    expect((await post(app(s), "/typing", { to: MOI, on: true })).status).toBe(204);
    expect((await post(app(s), "/typing", { to: MOI, on: false })).status).toBe(204);
    expect(s.typing).toEqual([`on:${MOI}`, `off:${MOI}`]);
    expect((await post(app(s), "/typing", { to: INCONNU, on: true })).status).toBe(403);
  });
});
