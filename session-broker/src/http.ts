import { timingSafeEqual } from "node:crypto";
import type { BrokerConfig } from "./config";
import type { InboundEvent } from "./inbound";
import type { Store } from "./store";

export type BrokerStatus = { lastPollOkAt: string | null; lastSendOkAt: string | null; sessionId: string };

export type HttpDeps = {
  cfg: BrokerConfig;
  status: BrokerStatus;
  store?: Store;
};

function autorise(req: Request, token: string): boolean {
  const h = req.headers.get("authorization") ?? "";
  const m = /^Bearer (.+)$/.exec(h);
  if (!m) return false;
  const a = Buffer.from(m[1]);
  const b = Buffer.from(token);
  return a.length === b.length && timingSafeEqual(a, b);
}

const json = (obj: unknown, status = 200) =>
  new Response(JSON.stringify(obj), { status, headers: { "Content-Type": "application/json" } });

const trame = (evt: InboundEvent) => new TextEncoder().encode(`event: message\ndata: ${JSON.stringify(evt)}\n\n`);

export function createServer(deps: HttpDeps) {
  const abonnes = new Set<ReadableStreamDefaultController<Uint8Array>>();

  function publish(evt: InboundEvent): void {
    for (const c of abonnes) {
      try { c.enqueue(trame(evt)); } catch { abonnes.delete(c); }
    }
  }

  async function fetch(req: Request): Promise<Response> {
    if (!autorise(req, deps.cfg.token)) return new Response("unauthorized", { status: 401 });
    const { pathname } = new URL(req.url);
    if (req.method === "GET" && pathname === "/health") {
      return json({ ok: true, ...deps.status });
    }
    if (req.method === "GET" && pathname === "/events" && deps.store) {
      const store = deps.store;
      let ctrl: ReadableStreamDefaultController<Uint8Array>;
      const stream = new ReadableStream<Uint8Array>({
        start(c) {
          ctrl = c;
          c.enqueue(new TextEncoder().encode(": ok\n\n"));
          for (const evt of store.pending()) c.enqueue(trame(evt));
          abonnes.add(c);
        },
        cancel() { abonnes.delete(ctrl); },
      });
      req.signal.addEventListener("abort", () => abonnes.delete(ctrl));
      return new Response(stream, { headers: { "Content-Type": "text/event-stream", "Cache-Control": "no-cache" } });
    }
    if (req.method === "POST" && pathname === "/ack" && deps.store) {
      const { id } = await req.json().catch(() => ({}));
      if (typeof id !== "string") return new Response("id requis", { status: 400 });
      deps.store.ack(id);
      return new Response(null, { status: 204 });
    }
    return new Response("not found", { status: 404 });
  }

  return { fetch, publish };
}
