import { describe, it, expect } from "bun:test";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { loadConfig } from "../src/config";

const ID = "05" + "a".repeat(64);

function cfgFile(obj: unknown): string {
  const d = mkdtempSync(join(tmpdir(), "sb-cfg-"));
  const p = join(d, "config.json");
  writeFileSync(p, JSON.stringify(obj));
  return p;
}

describe("loadConfig", () => {
  it("lève sans jeton", () => {
    expect(() => loadConfig(cfgFile({ port: 8091, allowFrom: [ID] }), {})).toThrow(/TIRON_SESSION_BROKER_TOKEN/);
  });
  it("lève sur allowFrom vide", () => {
    expect(() => loadConfig(cfgFile({ port: 8091, allowFrom: [] }), { TIRON_SESSION_BROKER_TOKEN: "t" })).toThrow(/allowFrom/);
  });
  it("lève sur un identifiant invalide", () => {
    expect(() => loadConfig(cfgFile({ port: 8091, allowFrom: ["05abc"] }), { TIRON_SESSION_BROKER_TOKEN: "t" })).toThrow(/identifiant/);
  });
  it("rend stateDir par défaut", () => {
    const c = loadConfig(cfgFile({ port: 8091, allowFrom: [ID] }), { TIRON_SESSION_BROKER_TOKEN: "t", HOME: "/h" });
    expect(c.stateDir).toBe("/h/.local/state/session-broker");
    expect(c.port).toBe(8091);
    expect(c.allowFrom).toEqual([ID]);
    expect(c.token).toBe("t");
  });
});
