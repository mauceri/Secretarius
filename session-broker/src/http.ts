import { timingSafeEqual } from "node:crypto";
import type { BrokerConfig } from "./config";

export type BrokerStatus = { lastPollOkAt: string | null; lastSendOkAt: string | null; sessionId: string };

export type HttpDeps = {
  cfg: BrokerConfig;
  status: BrokerStatus;
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

export function createServer(deps: HttpDeps) {
  return {
    async fetch(req: Request): Promise<Response> {
      if (!autorise(req, deps.cfg.token)) return new Response("unauthorized", { status: 401 });
      const { pathname } = new URL(req.url);
      if (req.method === "GET" && pathname === "/health") {
        return json({ ok: true, ...deps.status });
      }
      return new Response("not found", { status: 404 });
    },
  };
}
