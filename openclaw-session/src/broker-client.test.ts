import { describe, it, expect, afterEach } from "vitest";
import { createServer, type Server } from "node:http";
import { BrokerClient, BrokerNotAllowedError } from "./broker-client.js";

let srv: Server | null = null;
afterEach(() => { srv?.close(); srv = null; });

function serveur(handler: (req: any, res: any) => void): Promise<string> {
  return new Promise((ok) => {
    srv = createServer(handler).listen(0, "127.0.0.1", () => ok(`http://127.0.0.1:${(srv!.address() as any).port}`));
  });
}

describe("BrokerClient", () => {
  it("envoie le jeton et parse deux événements SSE", async () => {
    let auth = "";
    const url = await serveur((req, res) => {
      auth = req.headers.authorization;
      res.writeHead(200, { "Content-Type": "text/event-stream" });
      res.write(": ok\n\n");
      res.write('event: message\ndata: {"id":"a","from":"05x","timestamp":1,"text":"un","attachments":[]}\n\n');
      res.write('event: message\ndata: {"id":"b","from":"05x","timestamp":2,"text":"deux","attachments":[]}\n\n');
    });
    const ac = new AbortController();
    const ids: string[] = [];
    for await (const e of new BrokerClient(url, "jeton").events(ac.signal)) {
      ids.push(e.id);
      if (ids.length === 2) ac.abort();
    }
    expect(auth).toBe("Bearer jeton");
    expect(ids).toEqual(["a", "b"]);
  });
  it("send : 403 → BrokerNotAllowedError", async () => {
    const url = await serveur((_req, res) => { res.writeHead(403); res.end(); });
    await expect(new BrokerClient(url, "j").send("05x", "t")).rejects.toBeInstanceOf(BrokerNotAllowedError);
  });
  it("send : 200 → messageId", async () => {
    let corps = "";
    const url = await serveur((req, res) => {
      req.on("data", (c: Buffer) => (corps += c));
      req.on("end", () => { res.writeHead(200, { "Content-Type": "application/json" }); res.end('{"messageId":"h","timestamp":3}'); });
    });
    expect(await new BrokerClient(url, "j").send("05x", "t")).toEqual({ messageId: "h" });
    expect(JSON.parse(corps)).toEqual({ to: "05x", text: "t" });
  });
  it("ack et typing postent le bon corps", async () => {
    const recus: any[] = [];
    const url = await serveur((req, res) => {
      let c = ""; req.on("data", (d: Buffer) => (c += d));
      req.on("end", () => { recus.push([req.url, JSON.parse(c)]); res.writeHead(204); res.end(); });
    });
    const cl = new BrokerClient(url, "j");
    await cl.ack("a");
    await cl.typing("05x", true);
    expect(recus).toEqual([["/ack", { id: "a" }], ["/typing", { to: "05x", on: true }]]);
  });
  it("events : statut non 200 → erreur", async () => {
    const url = await serveur((_req, res) => { res.writeHead(401); res.end(); });
    const it = new BrokerClient(url, "j").events(new AbortController().signal)[Symbol.asyncIterator]();
    await expect(it.next()).rejects.toThrow(/401/);
  });
});
