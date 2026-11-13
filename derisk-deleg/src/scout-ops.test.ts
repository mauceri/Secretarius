import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, renameSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { CONSIGNE_RESUME, lireSource, resumerObfusque, tacheScout, verifierCourriel } from "./scout-ops.js";

// Faux scout-watcher : prend chaque tâche de pending/, la complète, la pose dans done/.
let ws: string;
let minuteur: ReturnType<typeof setInterval> | null = null;
let vues: any[] = [];

function watcher(completer: (d: any) => any) {
  const pending = join(ws, "tasks", "pending");
  const done = join(ws, "tasks", "done");
  mkdirSync(pending, { recursive: true });
  mkdirSync(done, { recursive: true });
  minuteur = setInterval(() => {
    for (const f of readdirSync(pending).filter((n) => n.endsWith(".json"))) {
      const d = JSON.parse(readFileSync(join(pending, f), "utf8"));
      vues.push(d);
      writeFileSync(join(pending, f), JSON.stringify(completer(d)));
      renameSync(join(pending, f), join(done, f));
    }
  }, 5);
}
const garde = (g: any) => (d: any) => ({ ...d, fetched_content: JSON.stringify(g) });
const deps = (extra: any = {}) => ({ workspace: ws, pollMs: 5, timeoutMs: 2000, ...extra });

beforeEach(() => { ws = mkdtempSync(join(tmpdir(), "scout-")); vues = []; });
afterEach(() => {
  if (minuteur) clearInterval(minuteur);
  minuteur = null;
  rmSync(ws, { recursive: true, force: true });
  vi.unstubAllGlobals();
});

describe("tacheScout", () => {
  it("dépose une tâche URL au format du watcher et rend le verdict du garde", async () => {
    watcher(garde({ blocked: false, risk: "low", clean_text: "Texte propre" }));
    const v = await tacheScout({ url: "https://example.org" }, deps());
    expect(v).toEqual({ status: "ok", risk: "low", clean_text: "Texte propre" });
    expect(vues[0].url_or_path).toBe("https://example.org");
  });

  it("dépose une tâche check_email", async () => {
    watcher(garde({ blocked: false, risk: "low", clean_text: "Bonjour" }));
    await tacheScout({ check_email: "Bonjour" }, deps());
    expect(vues[0].check_email).toBe("Bonjour");
    expect(vues[0].url_or_path).toBeUndefined();
  });

  it("rend l'erreur de récupération", async () => {
    watcher((d) => ({ ...d, fetch_error: "404" }));
    const v = await tacheScout({ url: "https://example.org/x" }, deps());
    expect(v).toEqual({ status: "error", error: "récupération impossible (HTTP 404)" });
  });

  it("expire si le watcher ne répond pas", async () => {
    const v = await tacheScout({ url: "https://example.org" }, deps({ timeoutMs: 50 }));
    expect(v).toEqual({ status: "timeout" });
  });
});

describe("lireSource", () => {
  it("contenu bloqué : motif rendu, aucun appel au modèle", async () => {
    watcher(garde({ blocked: true, reason: "injection détectée" }));
    const resumer = vi.fn();
    const out = await lireSource("https://example.org", deps({ resumer }));
    expect(out).toContain("injection détectée");
    expect(resumer).not.toHaveBeenCalled();
  });

  it("résume clean_text et signale un risque moyen", async () => {
    watcher(garde({ blocked: false, risk: "medium", clean_text: "Texte propre" }));
    const resumer = vi.fn(async () => "Un résumé.");
    const out = await lireSource("https://example.org", deps({ resumer }));
    expect(resumer).toHaveBeenCalledWith("Texte propre");
    expect(out).toContain("Attention");
    expect(out).toContain("Un résumé.");
  });

  it("watcher absent : Scout indisponible", async () => {
    const out = await lireSource("https://example.org", deps({ timeoutMs: 50 }));
    expect(out).toContain("Scout indisponible");
  });
});

describe("verifierCourriel", () => {
  it("rend le verdict sans résumer, texte balisé UNTRUSTED", async () => {
    watcher(garde({ blocked: false, risk: "low", clean_text: "Bonjour" }));
    const out = JSON.parse(await verifierCourriel("Bonjour", deps()));
    expect(out).toEqual({ blocked: false, risk: "low", clean_text: "<UNTRUSTED> Bonjour" });
  });
});

describe("resumerObfusque", () => {
  it("n'envoie que la consigne fixe et le texte balisé, sans outils, et retire <think>", async () => {
    const fetchSpy = vi.fn(async () => ({
      ok: true,
      json: async () => ({ choices: [{ message: { content: "<think>bla</think>\nRésumé." } }] }),
    }));
    vi.stubGlobal("fetch", fetchSpy);
    const out = await resumerObfusque("Texte de la page");
    expect(out).toBe("Résumé.");
    const corps = JSON.parse((fetchSpy.mock.calls[0] as any)[1].body);
    expect(corps.tools).toBeUndefined();
    expect(corps.messages).toEqual([
      { role: "system", content: CONSIGNE_RESUME },
      { role: "user", content: "<UNTRUSTED>\nTexte de la page\n</UNTRUSTED>" },
    ]);
  });
});
