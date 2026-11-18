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
    expect(contenu).toContain(`SESSION_BOT_MNEMONIC='${PHRASE}'\n`);
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

describe("secrets.env reste sourçable par bash", () => {
  it("set -a; . fichier rend exactement la phrase, sans rien exécuter ni afficher", () => {
    const p = join(mkdtempSync(join(tmpdir(), "sb-init-")), "secrets.env");
    writeFileSync(p, "AUTRE=1\n", { mode: 0o600 });
    // « echo » en 2e position : sans guillemets, bash l'exécuterait et afficherait la suite.
    initIdentity(p, () => "alpha echo gamma delta", idDe);
    const r = Bun.spawnSync(["bash", "-c", `set -eu; set -a; . '${p}'; set +a; printf '%s' "$SESSION_BOT_MNEMONIC"`]);
    expect(r.exitCode).toBe(0);
    expect(r.stderr.toString()).toBe("");
    expect(r.stdout.toString()).toBe("alpha echo gamma delta");
  });
  it("l'enveloppe bin/session-broker extrait la valeur sans guillemets", () => {
    const p = join(mkdtempSync(join(tmpdir(), "sb-init-")), "secrets.env");
    initIdentity(p, () => "alpha echo gamma delta", idDe);
    const cle = readFileSync(join(import.meta.dir, "..", "bin", "session-broker"), "utf8").split("\n").find((l) => l.startsWith("cle()"))!;
    const r = Bun.spawnSync(["bash", "-c", `f='${p}'; ${cle}; printf '%s' "$(cle SESSION_BOT_MNEMONIC)"`]);
    expect(r.stdout.toString()).toBe("alpha echo gamma delta");
  });
});
