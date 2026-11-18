// Plugin de canal « session » : Session Messenger par le broker local.
import { createChannelPluginBase, createChatChannelPlugin } from "openclaw/plugin-sdk/channel-core";
import { join } from "node:path";
import { homedir } from "node:os";
import { sleepWithAbort } from "openclaw/plugin-sdk/runtime-env";
import { BrokerClient } from "./broker-client.js";
import { boucleReception, traiterEvenement, Vus } from "./monitor.js";
import { deposerPiecesJointes } from "./depot.js";
import { creerDispatchTexte } from "./dispatch-sdk.js";
const section = (cfg) => (cfg?.channels?.session ?? {});
export function resolveSessionAccount(cfg, _accountId) {
    const s = section(cfg);
    const brokerUrl = typeof s.brokerUrl === "string" ? s.brokerUrl : "";
    const token = process.env.TIRON_SESSION_BROKER_TOKEN ?? "";
    return {
        accountId: "default",
        brokerUrl,
        allowFrom: Array.isArray(s.allowFrom) ? s.allowFrom.map(String) : [],
        dmPolicy: "allowlist",
        token,
        enabled: s.enabled !== false,
        configured: Boolean(brokerUrl),
    };
}
export function inspectSessionAccount(cfg, _accountId) {
    const s = section(cfg);
    return {
        enabled: s.enabled !== false,
        configured: typeof s.brokerUrl === "string" && s.brokerUrl.length > 0,
        tokenStatus: process.env.TIRON_SESSION_BROKER_TOKEN ? "available" : "missing",
    };
}
export const clientPour = (a) => new BrokerClient(a.brokerUrl, a.token);
export const sessionPlugin = createChatChannelPlugin({
    base: {
        ...createChannelPluginBase({
            id: "session",
            meta: { label: "Session", selectionLabel: "Session (broker local)", blurb: "Session Messenger par le broker local de Secretarius." },
            // Le cœur lit capabilities.tts sans garde : capabilities est obligatoire.
            capabilities: { chatTypes: ["direct"], media: false, reactions: false },
            config: {
                listAccountIds: () => ["default"],
                resolveAccount: resolveSessionAccount,
                inspectAccount: inspectSessionAccount,
            },
        }),
        gateway: {
            startAccount: async (ctx) => {
                const account = ctx.account;
                const client = clientPour(account);
                const log = (l) => ctx.log?.info?.(l);
                const receptionDir = process.env.TIRON_RECEPTION_DIR || join(homedir(), "Documents", "Arbath", "Réception");
                const seen = new Vus(join(homedir(), ".local", "state", "openclaw-session", "vus.json"));
                const dispatchTexte = creerDispatchTexte({ cfg: ctx.cfg, account, client, channelRuntime: ctx.channelRuntime, runtime: ctx.runtime, log });
                log(`[session] démarrage (${account.brokerUrl})`);
                await boucleReception({
                    abortSignal: ctx.abortSignal,
                    events: (signal) => client.events(signal),
                    traiter: async (evt) => {
                        await traiterEvenement(evt, {
                            allowFrom: account.allowFrom, seen, receptionDir, depot: deposerPiecesJointes,
                            dispatchTexte, repondre: async (to, text) => { await client.send(to, text); }, ack: (id) => client.ack(id),
                        });
                    },
                    sleep: (ms, signal) => sleepWithAbort(ms, signal),
                    log,
                });
            },
        },
        heartbeat: {
            sendTyping: async ({ cfg, to }) => { await clientPour(resolveSessionAccount(cfg)).typing(String(to).replace(/^session:/, ""), true); },
            clearTyping: async ({ cfg, to }) => { await clientPour(resolveSessionAccount(cfg)).typing(String(to).replace(/^session:/, ""), false); },
        },
    },
    security: {
        dm: {
            channelKey: "session",
            defaultPolicy: "allowlist",
            resolvePolicy: (a) => a.dmPolicy,
            resolveAllowFrom: (a) => a.allowFrom,
        },
    },
    threading: { topLevelReplyToMode: "off" },
    outbound: {
        base: { deliveryMode: "direct", textChunkLimit: 4000, chunkerMode: "text" },
        attachedResults: {
            channel: "session",
            sendText: async (ctx) => {
                const a = resolveSessionAccount(ctx.cfg, ctx.accountId);
                return clientPour(a).send(String(ctx.to).replace(/^session:/, ""), String(ctx.text));
            },
        },
    },
});
