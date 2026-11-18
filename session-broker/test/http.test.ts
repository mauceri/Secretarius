import { describe, it, expect } from "bun:test";
import { createServer, type HttpDeps } from "../src/http";

const ID = "05" + "a".repeat(64);
function deps(): HttpDeps {
  return {
    cfg: { port: 0, allowFrom: [ID], stateDir: "/tmp/x", token: "secret", maxAttachmentBytes: 10 * 1024 * 1024 },
    status: { lastPollOkAt: null, lastSendOkAt: null, sessionId: "05bot" },
  } as HttpDeps;
}
const req = (path: string, init: RequestInit = {}) => new Request(`http://127.0.0.1${path}`, init);

describe("http", () => {
  it("401 sans en-tête", async () => {
    expect((await createServer(deps()).fetch(req("/health"))).status).toBe(401);
  });
  it("401 avec un mauvais jeton", async () => {
    const r = await createServer(deps()).fetch(req("/health", { headers: { Authorization: "Bearer faux" } }));
    expect(r.status).toBe(401);
  });
  it("200 /health avec le bon jeton", async () => {
    const r = await createServer(deps()).fetch(req("/health", { headers: { Authorization: "Bearer secret" } }));
    expect(r.status).toBe(200);
    expect(await r.json()).toEqual({ ok: true, lastPollOkAt: null, lastSendOkAt: null, sessionId: "05bot" });
  });
  it("404 sur route inconnue", async () => {
    const r = await createServer(deps()).fetch(req("/nope", { headers: { Authorization: "Bearer secret" } }));
    expect(r.status).toBe(404);
  });
});
