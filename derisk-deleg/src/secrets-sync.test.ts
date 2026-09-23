import { describe, expect, it, vi } from "vitest";
import { parseEnv, diffEnv, applyEnvDiff, validateTelegramToken, EXPECTED_TELEGRAM_BOT } from "./secrets-sync.js";

describe("parseEnv", () => {
  it("lit des valeurs sans guillemets", () => {
    expect(parseEnv("FOO=bar\nBAZ=qux\n")).toEqual({ FOO: "bar", BAZ: "qux" });
  });

  it("retire un seul niveau de guillemets doubles", () => {
    expect(parseEnv('FOO="bar"\n')).toEqual({ FOO: "bar" });
  });

  it("retire un seul niveau de guillemets simples", () => {
    expect(parseEnv("FOO='bar'\n")).toEqual({ FOO: "bar" });
  });

  it("ignore les lignes commentées", () => {
    expect(parseEnv("#FOO=bar\nBAZ=qux\n")).toEqual({ BAZ: "qux" });
  });

  it("ignore les lignes vides et sans '='", () => {
    expect(parseEnv("\nFOO\nBAZ=qux\n")).toEqual({ BAZ: "qux" });
  });

  it("conserve un '=' dans la valeur", () => {
    expect(parseEnv("FOO=a=b=c\n")).toEqual({ FOO: "a=b=c" });
  });
});

describe("diffEnv", () => {
  it("ne retient que les clés présentes dans les deux et différentes", () => {
    const source = { A: "new", B: "same", C: "onlyInSource" };
    const target = { A: "old", B: "same", D: "onlyInTarget" };
    expect(diffEnv(source, target)).toEqual([{ key: "A", oldValue: "old", newValue: "new" }]);
  });

  it("retourne un tableau vide si rien ne diffère", () => {
    expect(diffEnv({ A: "1" }, { A: "1" })).toEqual([]);
  });
});

describe("applyEnvDiff", () => {
  it("remplace la ligne ciblée sans toucher au reste, sans ajouter de guillemets", () => {
    const content = "A=old\nB=untouched\n";
    const out = applyEnvDiff(content, [{ key: "A", oldValue: "old", newValue: "new" }]);
    expect(out).toBe("A=new\nB=untouched\n");
  });

  it("n'ajoute pas de ligne pour une clé absente du fichier cible", () => {
    const content = "A=old\n";
    const out = applyEnvDiff(content, [{ key: "Z", oldValue: "x", newValue: "y" }]);
    expect(out).toBe("A=old\n");
  });
});

describe("validateTelegramToken", () => {
  it("accepte un token dont le username getMe correspond", async () => {
    const fetchImpl = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ ok: true, result: { username: "secretarius_tiron_bot" } }),
    });
    const res = await validateTelegramToken("tok", "secretarius_tiron_bot", fetchImpl as any);
    expect(res.ok).toBe(true);
  });

  it("refuse un token dont le username getMe ne correspond pas (piège d'hier)", async () => {
    const fetchImpl = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ ok: true, result: { username: "secretarius1789_bot" } }),
    });
    const res = await validateTelegramToken("tok", "secretarius_tiron_bot", fetchImpl as any);
    expect(res.ok).toBe(false);
    expect(res.actualUsername).toBe("secretarius1789_bot");
  });

  it("refuse si l'appel HTTP échoue", async () => {
    const fetchImpl = vi.fn().mockResolvedValue({ ok: false, status: 401 });
    const res = await validateTelegramToken("tok", "secretarius_tiron_bot", fetchImpl as any);
    expect(res.ok).toBe(false);
  });

  it("refuse si fetch lève une exception (réseau)", async () => {
    const fetchImpl = vi.fn().mockRejectedValue(new Error("boom"));
    const res = await validateTelegramToken("tok", "secretarius_tiron_bot", fetchImpl as any);
    expect(res.ok).toBe(false);
  });
});

describe("EXPECTED_TELEGRAM_BOT", () => {
  it("connaît sanroque (dev) et santiago (prod)", () => {
    expect(EXPECTED_TELEGRAM_BOT.sanroque).toBe("secretarius_tiron_bot");
    expect(EXPECTED_TELEGRAM_BOT.santiago).toBe("secretarius1789_bot");
  });
});
