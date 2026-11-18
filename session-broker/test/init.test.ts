import { describe, it, expect } from "bun:test";
import { mkdtempSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { initIdentity } from "../src/init";

// Phrase de test fixe (identité jetable, jamais utilisée sur le réseau).
const PHRASE = "fictif mot deux trois quatre cinq six sept huit neuf dix onze douze";
const gen = () => PHRASE;
const idDe = (_m: string) => "05" + "c".repeat(64);

describe("initIdentity", () => {
  it("ajoute les deux clés, rend seulement l'identifiant, crée en 0600", () => {
    const p = join(mkdtempSync(join(tmpdir(), "sb-init-")), "secrets.env");
    const r = initIdentity(p, gen, idDe);
    const contenu = readFileSync(p, "utf8");
    expect(contenu).toContain(`SESSION_BOT_MNEMONIC=${PHRASE}\n`);
    expect(contenu).toMatch(/^TIRON_SESSION_BROKER_TOKEN=[0-9a-f]{64}$/m);
    expect(r).toEqual({ sessionId: "05" + "c".repeat(64) });
    expect(JSON.stringify(r)).not.toContain("fictif");
    expect(statSync(p).mode & 0o777).toBe(0o600);
  });
  it("conserve le contenu et le mode existants", () => {
    const p = join(mkdtempSync(join(tmpdir(), "sb-init-")), "secrets.env");
    writeFileSync(p, "AUTRE=1", { mode: 0o600 });
    initIdentity(p, gen, idDe);
    const c = readFileSync(p, "utf8");
    expect(c.startsWith("AUTRE=1\n")).toBe(true);
    expect(statSync(p).mode & 0o777).toBe(0o600);
  });
  it("refuse une seconde fois et ne modifie rien", () => {
    const p = join(mkdtempSync(join(tmpdir(), "sb-init-")), "secrets.env");
    initIdentity(p, gen, idDe);
    const avant = readFileSync(p, "utf8");
    expect(() => initIdentity(p, gen, idDe)).toThrow(/existe déjà/);
    expect(readFileSync(p, "utf8")).toBe(avant);
  });
});
