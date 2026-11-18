import { describe, it, expect, afterEach } from "bun:test";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createServer } from "../src/http";
import { Store } from "../src/store";

const MOI = "05" + "a".repeat(64);
let srv: ReturnType<typeof Bun.serve> | null = null;
afterEach(() => { srv?.stop(true); srv = null; });

function demarrer(keepaliveMs?: number) {
  const d = mkdtempSync(join(tmpdir(), "sb-sse-"));
  const store = new Store(join(d, "broker.sqlite"));
  const cfg = { port: 0, allowFrom: [MOI], stateDir: d, token: "t", maxAttachmentBytes: 10 };
  const app = createServer({ cfg, status: { lastPollOkAt: null, lastSendOkAt: null, sessionId: "05bot" }, store, keepaliveMs } as any);
  srv = Bun.serve({ port: 0, hostname: "127.0.0.1", fetch: app.fetch });
  return { store, app, base: `http://127.0.0.1:${srv.port}` };
}
const H = { Authorization: "Bearer t" };
const evt = (id: string) => ({ id, from: MOI, timestamp: 1, text: "x", attachments: [] });

async function premierEvenement(base: string): Promise<any> {
  const ac = new AbortController();
  const r = await fetch(`${base}/events`, { headers: H, signal: ac.signal });
  const reader = r.body!.getReader();
  let buf = "";
  while (true) {
    const { value } = await reader.read();
    buf += new TextDecoder().decode(value);
    const m = /data: (.+)\n\n/.exec(buf);
    if (m) { ac.abort(); return JSON.parse(m[1]); }
  }
}

describe("SSE", () => {
  it("envoie un commentaire de maintien périodique", async () => {
    const { base } = demarrer(30);
    const ac = new AbortController();
    const r = await fetch(`${base}/events`, { headers: H, signal: ac.signal });
    const reader = r.body!.getReader();
    let buf = "";
    while (!buf.includes(": ping")) buf += new TextDecoder().decode((await reader.read()).value);
    ac.abort();
    expect(buf).toContain(": ping\n\n");
  });
  it("/events sans jeton : 401", async () => {
    const { base } = demarrer();
    expect((await fetch(`${base}/events`)).status).toBe(401);
  });
  it("rejoue un événement non acquitté, plus après /ack", async () => {
    const { store, base, app } = demarrer();
    store.enqueue(evt("e1"));
    expect((await premierEvenement(base)).id).toBe("e1");
    expect((await premierEvenement(base)).id).toBe("e1");
    const r = await fetch(`${base}/ack`, { method: "POST", headers: { ...H, "Content-Type": "application/json" }, body: JSON.stringify({ id: "e1" }) });
    expect(r.status).toBe(204);
    store.enqueue(evt("e2"));
    expect((await premierEvenement(base)).id).toBe("e2");
  });
  it("diffuse un événement publié après la connexion", async () => {
    const { base, app } = demarrer();
    const p = premierEvenement(base);
    await Bun.sleep(50);
    app.publish(evt("e3"));
    expect((await p).id).toBe("e3");
  });
});
