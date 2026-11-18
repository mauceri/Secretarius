import { readFileSync } from "node:fs";
import { join } from "node:path";

export type BrokerConfig = {
  port: number;
  allowFrom: string[];
  stateDir: string;
  token: string;
  maxAttachmentBytes: number;
};

const ID_SESSION = /^05[0-9a-f]{64}$/;

export function loadConfig(path: string, env: Record<string, string | undefined>): BrokerConfig {
  const brut = JSON.parse(readFileSync(path, "utf8"));
  const token = env.TIRON_SESSION_BROKER_TOKEN;
  if (!token) throw new Error("TIRON_SESSION_BROKER_TOKEN absent");
  const allowFrom: unknown = brut.allowFrom;
  if (!Array.isArray(allowFrom) || allowFrom.length === 0) throw new Error("allowFrom vide ou absent");
  for (const id of allowFrom) {
    if (typeof id !== "string" || !ID_SESSION.test(id)) throw new Error(`identifiant Session invalide dans allowFrom : ${String(id).slice(0, 8)}…`);
  }
  if (!Number.isInteger(brut.port) || brut.port <= 0) throw new Error("port invalide");
  return {
    port: brut.port,
    allowFrom: allowFrom as string[],
    stateDir: brut.stateDir ?? join(env.HOME ?? "", ".local", "state", "session-broker"),
    token,
    maxAttachmentBytes: brut.maxAttachmentBytes ?? 10 * 1024 * 1024,
  };
}
