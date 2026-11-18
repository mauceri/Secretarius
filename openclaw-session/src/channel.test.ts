import { describe, it, expect } from "vitest";
import { resolveSessionAccount, inspectSessionAccount } from "./channel.js";

const ID = "05" + "a".repeat(64);

describe("compte session", () => {
  it("resolveSessionAccount lit allowFrom, brokerUrl et le jeton de l'environnement", () => {
    process.env.TIRON_SESSION_BROKER_TOKEN = "jeton-test";
    const a = resolveSessionAccount({ channels: { session: { enabled: true, brokerUrl: "http://127.0.0.1:8091", dmPolicy: "allowlist", allowFrom: [ID] } } } as any);
    expect(a).toMatchObject({ accountId: "default", brokerUrl: "http://127.0.0.1:8091", allowFrom: [ID], dmPolicy: "allowlist", token: "jeton-test", enabled: true, configured: true });
  });
  it("inspect : configured false sans brokerUrl, jamais le jeton", () => {
    process.env.TIRON_SESSION_BROKER_TOKEN = "jeton-test";
    const r = inspectSessionAccount({ channels: { session: { enabled: true, allowFrom: [ID] } } } as any);
    expect(r.configured).toBe(false);
    expect(JSON.stringify(r)).not.toContain("jeton-test");
  });
  it("inspect : statut du jeton available/missing", () => {
    process.env.TIRON_SESSION_BROKER_TOKEN = "jeton-test";
    const cfg = { channels: { session: { enabled: true, brokerUrl: "http://127.0.0.1:8091", allowFrom: [ID] } } } as any;
    expect(inspectSessionAccount(cfg)).toMatchObject({ configured: true, tokenStatus: "available" });
    delete process.env.TIRON_SESSION_BROKER_TOKEN;
    expect(inspectSessionAccount(cfg)).toMatchObject({ tokenStatus: "missing" });
  });
});
