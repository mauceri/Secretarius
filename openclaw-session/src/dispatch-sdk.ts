// Transmission d'un message texte Session à l'agent lié (main), calquée sur le
// plugin @openclaw/signal@2026.8.2 (dist/monitor-*.js, handleSignalInboundMessage),
// réduite aux messages directs : ni groupes, ni réactions, ni approbations natives.
import { resolveAgentRoute, resolveInboundLastRouteSessionKey } from "openclaw/plugin-sdk/routing";
import { buildChannelInboundEventContext, formatInboundEnvelope, runChannelInboundEvent } from "openclaw/plugin-sdk/channel-inbound";
import { createChannelMessageReplyPipeline } from "openclaw/plugin-sdk/channel-outbound";
import { createChannelIngressResolver, defineStableChannelIngressIdentity } from "openclaw/plugin-sdk/channel-ingress-runtime";
import type { BrokerClient, InboundEvent } from "./broker-client.js";
import type { SessionAccount } from "./channel.js";

const ID_SESSION = /^(?:session:)?(05[0-9a-f]{64})$/;

const identite = defineStableChannelIngressIdentity({
  key: "session-id",
  normalizeEntry: (e: string) => ID_SESSION.exec(e.trim())?.[1] ?? null,
  normalizeSubject: (v: string) => v,
  sensitivity: "pii",
} as any);

export type DispatchParams = {
  cfg: any;
  account: SessionAccount;
  client: BrokerClient;
  channelRuntime: any;
  runtime: any;
  log(line: string): void;
};

export function creerDispatchTexte(p: DispatchParams): (evt: InboundEvent) => Promise<void> {
  const { cfg, account, client } = p;
  return async (evt) => {
    const route: any = resolveAgentRoute({ cfg, channel: "session", accountId: account.accountId, peer: { kind: "direct", id: evt.from } } as any);
    const ingress: any = await createChannelIngressResolver({
      channelId: "session",
      accountId: account.accountId,
      identity: identite,
      cfg,
      useDefaultPairingStore: false,
    } as any).message({
      subject: { stableId: evt.from },
      conversation: { kind: "direct", id: evt.from },
      contextBinding: { agentId: route.agentId, sessionKey: route.sessionKey, messageId: evt.id, inboundEventKind: "user_request" },
      dmPolicy: "allowlist",
      allowFrom: account.allowFrom,
      command: { allowTextCommands: true, directGroupAllowFrom: "effective" },
    } as any);
    if (!ingress?.senderAccess?.allowed) {
      p.log(`session : message de ${evt.from.slice(0, 8)} refusé par le contrôle d'accès`);
      return;
    }
    const to = `session:${evt.from}`;
    const body = formatInboundEnvelope({
      channel: "Session",
      from: evt.from.slice(0, 8),
      timestamp: evt.timestamp,
      body: evt.text,
      chatType: "direct",
      sender: { name: "", id: evt.from.slice(0, 8) },
      envelope: undefined,
    } as any);
    const ctxPayload = (p.channelRuntime?.inbound?.buildContext ?? buildChannelInboundEventContext)({
      channel: "session",
      messageId: evt.id,
      timestamp: evt.timestamp,
      from: to,
      sender: { id: evt.from },
      conversation: { kind: "direct", id: evt.from, label: evt.from.slice(0, 8) },
      route: { agentId: route.agentId, dmScope: route.dmScope, accountId: route.accountId, routeSessionKey: route.sessionKey },
      reply: { to },
      message: { body, bodyForAgent: evt.text, rawBody: evt.text, commandBody: evt.text },
      sessionTranscript: { historyLimit: 0 },
      access: { commands: { authorized: Boolean(ingress?.commandAccess?.authorized ?? true) } },
      channelIngress: [ingress],
    } as any);
    const { onModelSelected, typingCallbacks, ...replyPipeline } = createChannelMessageReplyPipeline({
      cfg,
      agentId: route.agentId,
      channel: "session",
      accountId: route.accountId,
      typing: {
        start: async () => { await client.typing(evt.from, true); },
        onStartError: (err: unknown) => p.log(`session : saisie non signalée (${String(err)})`),
      },
    } as any) as any;
    const sessionKeyLast = resolveInboundLastRouteSessionKey({ route, sessionKey: route.sessionKey } as any);
    await runChannelInboundEvent({
      channel: "session",
      accountId: route.accountId,
      raw: evt,
      adapter: {
        ingest: () => ({ id: evt.id, timestamp: evt.timestamp, rawText: evt.text, raw: evt }),
        resolveTurn: () => ({
          cfg,
          channel: "session",
          accountId: route.accountId,
          route: { agentId: route.agentId, sessionKey: route.sessionKey },
          ctxPayload,
          dispatchReplyFromConfig: p.channelRuntime?.reply?.dispatchReplyFromConfig,
          record: {
            updateLastRoute: { sessionKey: sessionKeyLast, channel: "session", to: evt.from, accountId: route.accountId },
            onRecordError: (err: unknown) => p.log(`session : métadonnées de session non mises à jour (${String(err)})`),
          },
          dispatcherOptions: { ...replyPipeline, typingCallbacks },
          delivery: {
            deliver: async (payload: any) => {
              const text = String(payload?.text ?? "").trim();
              if (text) await client.send(evt.from, text);
            },
            onError: (err: unknown, info: any) => p.log(`session : réponse ${info?.kind ?? ""} non livrée (${String(err)})`),
          },
          sessionInitRetry: { delaysMs: [] },
          replyOptions: { onModelSelected, disableBlockStreaming: true },
        }),
      },
    } as any);
    await client.typing(evt.from, false).catch(() => {});
  };
}
