import { timingSafeEqual } from "node:crypto";
import type { BrokerConfig } from "./config";
import type { InboundEvent } from "./inbound";
import type { Store } from "./store";
import type { SessionLike } from "./inbound";
import { NotAllowedError, sendText, setTyping } from "./outbound";

export type BrokerStatus = { lastPollOkAt: string | null; lastSendOkAt: string | null; sessionId: string };

export type HttpDeps = {
  cfg: BrokerConfig;
  status: BrokerStatus;
  store?: Store;
  session?: SessionLike;
  sleep?: (ms: number) => Promise<void>;
  log?: (line: string) => void;
  keepaliveMs?: number;
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
      let minuterie: ReturnType<typeof setInterval> | undefined;
      const fin = () => { abonnes.delete(ctrl); if (minuterie) clearInterval(minuterie); };
      const stream = new ReadableStream<Uint8Array>({
        start(c) {
          ctrl = c;
          // Commentaire SSE périodique : garde la connexion ouverte (délais d'inactivité).
          minuterie = setInterval(() => {
            try { c.enqueue(new TextEncoder().encode(": ping\n\n")); } catch { fin(); }
          }, deps.keepaliveMs ?? 15000);
          c.enqueue(new TextEncoder().encode(": ok\n\n"));
          for (const evt of store.pending()) c.enqueue(trame(evt));
          abonnes.add(c);
        },
        cancel() { fin(); },
      });
      req.signal.addEventListener("abort", fin);
      return new Response(stream, { headers: { "Content-Type": "text/event-stream", "Cache-Control": "no-cache" } });
    }
    if (req.method === "POST" && pathname === "/ack" && deps.store) {
      const { id } = await req.json().catch(() => ({}));
      if (typeof id !== "string") return new Response("id requis", { status: 400 });
      deps.store.ack(id);
      return new Response(null, { status: 204 });
    }
    if (req.method === "POST" && (pathname === "/send" || pathname === "/typing") && deps.session) {
      const body: any = await req.json().catch(() => ({}));
      const od = { session: deps.session, cfg: deps.cfg, sleep: deps.sleep, log: deps.log ?? (() => {}) };
      if (typeof body.to !== "string") return new Response("to requis", { status: 400 });
      try {
        if (pathname === "/typing") {
          await setTyping(body.to, Boolean(body.on), od);
          return new Response(null, { status: 204 });
        }
        if (typeof body.text !== "string" || !body.text) return new Response("text requis", { status: 400 });
        const r = await sendText(body.to, body.text, od);
        deps.status.lastSendOkAt = new Date().toISOString();
        return json(r);
      } catch (e) {
        if (e instanceof NotAllowedError) return new Response("destinataire non autorisé", { status: 403 });
        return new Response("échec d'envoi", { status: 502 });
      }
    }
    return new Response("not found", { status: 404 });
  }

  return { fetch, publish };
}
