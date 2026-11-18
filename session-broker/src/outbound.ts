import type { BrokerConfig } from "./config";
import { court, type SessionLike } from "./inbound";

export class NotAllowedError extends Error {}

export type OutboundDeps = {
  session: SessionLike;
  cfg: BrokerConfig;
  sleep?: (ms: number) => Promise<void>;
  log: (line: string) => void;
};

const DELAIS = [1000, 3000];

export async function sendText(to: string, text: string, deps: OutboundDeps): Promise<{ messageId: string; timestamp: number }> {
  if (!deps.cfg.allowFrom.includes(to)) throw new NotAllowedError(`destinataire hors liste blanche : ${court(to)}`);
  const sleep = deps.sleep ?? ((ms: number) => Bun.sleep(ms));
  let derniere: unknown;
  for (let essai = 0; essai <= DELAIS.length; essai++) {
    try {
      const r = await deps.session.sendMessage({ to, text });
      deps.log(`envoyé à ${court(to)} : ${text.length} car.`);
      return { messageId: r.messageHash, timestamp: r.timestamp };
    } catch (e) {
      derniere = e;
      deps.log(`échec d'envoi à ${court(to)} (essai ${essai + 1}) : ${e instanceof Error ? e.message : String(e)}`);
      if (essai < DELAIS.length) await sleep(DELAIS[essai]);
    }
  }
  throw derniere instanceof Error ? derniere : new Error(String(derniere));
}

export async function setTyping(to: string, on: boolean, deps: OutboundDeps): Promise<void> {
  if (!deps.cfg.allowFrom.includes(to)) throw new NotAllowedError(`destinataire hors liste blanche : ${court(to)}`);
  if (on) await deps.session.showTypingIndicator({ conversation: to });
  else await deps.session.hideTypingIndicator({ conversation: to });
}
