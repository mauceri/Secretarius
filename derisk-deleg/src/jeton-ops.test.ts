import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { chmodSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { commandeJeton } from "./jeton-ops.js";

// Faux porte-jeton : consigne ses arguments, imite les sorties du vrai.
let dir: string;
let bin: string;

beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), "jeton-"));
  bin = join(dir, "porte-jeton");
  writeFileSync(
    bin,
    `#!/bin/sh
printf '%s\\n' "$@" > "${dir}/args"
case "$1" in
  ajouter) [ "$2" = pris ] && { echo "préfixe déjà présent : pris" >&2; exit 1; }
           echo "Jeton de l'agent « $2 » (affiché une seule fois) :"; echo SECRET123 ;;
  retirer) [ "$2" = inconnu ] && { echo "Agent inconnu : inconnu" >&2; exit 1; }
           echo "Agent « $2 » retiré." ;;
  lister)  [ -f "${dir}/vide" ] || { echo veille; echo orchestrateur; } ;;
esac
`,
  );
  chmodSync(bin, 0o755);
});

afterEach(() => rmSync(dir, { recursive: true, force: true }));

const args = () => readFileSync(join(dir, "args"), "utf8").trim().split("\n");

describe("commandeJeton", () => {
  it("ajouter : renvoie le jeton", () => {
    const r = commandeJeton("ajouter redaction", bin);
    expect(r).toContain("SECRET123");
    expect(args()).toEqual(["ajouter", "redaction"]);
  });

  it("retirer : confirme le retrait", () => {
    expect(commandeJeton("retirer veille", bin)).toBe("Agent « veille » retiré.");
    expect(args()).toEqual(["retirer", "veille"]);
  });

  it("lister : un préfixe par ligne", () => {
    expect(commandeJeton("lister", bin)).toBe("veille\norchestrateur");
  });

  it("lister sans agent : message explicite", () => {
    writeFileSync(join(dir, "vide"), "");
    expect(commandeJeton("lister", bin)).toBe("Aucun agent.");
  });

  it("erreur de porte-jeton : son message est relayé", () => {
    expect(commandeJeton("ajouter pris", bin)).toBe("préfixe déjà présent : pris");
    expect(commandeJeton("retirer inconnu", bin)).toBe("Agent inconnu : inconnu");
  });

  it("porte absente : rien n'est lancé", () => {
    expect(commandeJeton("lister", join(dir, "absent"))).toBe("Porte absente sur cette machine.");
  });

  it("usage invalide : aide, rien n'est lancé", () => {
    const aide = "Usage : /jeton ajouter <préfixe> | /jeton retirer <préfixe> | /jeton lister";
    for (const a of ["", "supprimer veille", "ajouter", "lister veille", "ajouter a b"]) {
      expect(commandeJeton(a, bin)).toBe(aide);
    }
    expect(() => readFileSync(join(dir, "args"))).toThrow();
  });

  it("l'argument n'est jamais interprété par un shell", () => {
    commandeJeton("ajouter $(touch_pwned)", bin);
    expect(args()).toEqual(["ajouter", "$(touch_pwned)"]);
  });
});
