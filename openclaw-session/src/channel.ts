// Plugin de canal « session » : Session Messenger par le broker local.
import { createChannelPluginBase, createChatChannelPlugin } from "openclaw/plugin-sdk/channel-core";
import { BrokerClient } from "./broker-client.js";

export type SessionAccount = {
  accountId: "default";
  brokerUrl: string;
  allowFrom: string[];
  dmPolicy: "allowlist";
  token: string;
  enabled: boolean;
  configured: boolean;
};

const section = (cfg: any) => (cfg?.channels?.session ?? {}) as Record<string, any>;

export function resolveSessionAccount(cfg: any, _accountId?: string | null): SessionAccount {
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

export function inspectSessionAccount(cfg: any, _accountId?: string | null) {
  const s = section(cfg);
  return {
    enabled: s.enabled !== false,
    configured: typeof s.brokerUrl === "string" && s.brokerUrl.length > 0,
    tokenStatus: process.env.TIRON_SESSION_BROKER_TOKEN ? "available" : "missing",
  };
}

export const clientPour = (a: SessionAccount) => new BrokerClient(a.brokerUrl, a.token);

export const sessionPlugin = createChatChannelPlugin<SessionAccount>({
  base: {
    ...createChannelPluginBase<SessionAccount>({
      id: "session",
      config: {
        listAccountIds: () => ["default"],
        resolveAccount: resolveSessionAccount,
        inspectAccount: inspectSessionAccount,
      } as any,
    }),
  } as any,
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
    base: { deliveryMode: "direct", textChunkLimit: 4000, chunkerMode: "text" } as any,
    attachedResults: {
      channel: "session",
      sendText: async (ctx: any) => {
        const a = resolveSessionAccount(ctx.cfg, ctx.accountId);
        return clientPour(a).send(String(ctx.to).replace(/^session:/, ""), String(ctx.text));
      },
    },
  },
});
