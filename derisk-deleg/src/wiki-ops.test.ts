import { describe, expect, it } from "vitest";
import { fetchWikiOpJson, formatWikiResult, runWikiOp } from "./wiki-ops.js";

describe("formatWikiResult", () => {
  it("query (regime full) : renvoie la synthèse verbatim", () => {
    expect(formatWikiResult("query", { synthesis: "# GPU TEE\n…", references: ["c-x"] }, "full"))
      .toBe("# GPU TEE\n…");
  });
  it("query : erreur surfacée verbatim", () => {
    expect(formatWikiResult("query", { error: "index vide" })).toBe("index vide");
  });
  it("capture : liste les fichiers", () => {
    expect(formatWikiResult("capture", { files: ["a.url", "b.url"] }))
      .toBe("Capturé : a.url, b.url (en file d'attente pour ingestion).");
  });
  it("ingest : mappe le status", () => {
    expect(formatWikiResult("ingest", { status: "launched", queued: 3 }))
      .toBe("Ingestion lancée en arrière-plan.");
    expect(formatWikiResult("ingest", { status: "nothing_to_do", queued: 0 }))
      .toBe("Rien à ingérer.");
    expect(formatWikiResult("ingest", { status: "already_running", queued: 2 }))
      .toBe("Ingestion déjà en cours.");
  });
  it("status : rend l'état sobrement", () => {
    expect(formatWikiResult("status", { running: true, last_run: null, pending: 4, blocked_files: [] }))
      .toBe("Ingestion en cours. En attente : 4. Bloqués : 0.");
    expect(formatWikiResult("status", { running: false, last_run: { ingested: 2, errors: 0 }, pending: 0, blocked_files: ["x.url"] }))
      .toBe("Ingestion à l'arrêt (dernier run : 2 ingéré(s), 0 erreur(s)). En attente : 0. Bloqués : 1.");
  });
  it("tags : joint la liste", () => {
    expect(formatWikiResult("tags", { tags: ["gpu", "tee"] })).toBe("Tags : gpu, tee.");
  });
  it("search : liste numérotée titre + extrait", () => {
    expect(formatWikiResult("search", { results: [
      { title: "Titre A", excerpt: "extrait A" },
      { title: "Titre B", excerpt: "extrait B" },
    ] })).toBe("1. Titre A\n   extrait A\n\n2. Titre B\n   extrait B");
  });
  it("search : liste vide", () => {
    expect(formatWikiResult("search", { results: [] })).toBe("Aucun résultat.");
  });
  it("kb_update : succès (status ok), async (launched) et erreur", () => {
    expect(formatWikiResult("kb_update", { status: "ok", clustering: "c1" }))
      .toBe("Base de connaissances mise à jour.");
    expect(formatWikiResult("kb_update", { status: "launched" }))
      .toBe("Mise à jour de la base lancée en arrière-plan.");
    expect(formatWikiResult("kb_update", { status: "error", reason: "clusterings/ introuvable" }))
      .toBe("clusterings/ introuvable");
  });
  it("delete_preview : liste ce qui serait affecté", () => {
    expect(formatWikiResult("delete_preview", { status: "ok", slug: "src-a", affected: ["src-a", "c-related"] }))
      .toBe("2 page(s) seraient affectées :\nsrc-a\nc-related");
  });
  it("delete : liste ce qui a été affecté", () => {
    expect(formatWikiResult("delete", { status: "ok", slug: "src-a", affected: ["src-a", "c-related"] }))
      .toBe("2 page(s) affectées :\nsrc-a\nc-related");
  });
  it("delete : liste vide", () => {
    expect(formatWikiResult("delete", { status: "ok", slug: "src-a", affected: [] }))
      .toBe("Aucune page affectée.");
  });
  it("delete_preview : erreur (slug introuvable) surfacée verbatim", () => {
    expect(formatWikiResult("delete_preview", { error: "Page introuvable pour le slug 'src-x'" }))
      .toBe("Page introuvable pour le slug 'src-x'");
  });
  it("erreur vide → ne renvoie pas un message vide (retombe sur l'op)", () => {
    expect(formatWikiResult("query", { error: "", synthesis: "# X" }, "full")).toBe("# X");
  });
  it("erreur générique inconnue → message par défaut", () => {
    expect(formatWikiResult("query", {})).toBe("Réponse wiki vide ou inattendue.");
  });
  it("query (regime brief, défaut) : résumé + chemin de la note dans le coffre", () => {
    expect(formatWikiResult("query", {
      synthesis: "# X", brief: "Résumé.", history_path: "Wiki_LM/historique/20260908-politiciens.md",
    })).toBe("Résumé.\n\nWiki_LM/historique/20260908-politiciens.md");
  });
  it("query (regime brief) : brief vide → chemin seul", () => {
    expect(formatWikiResult("query", { brief: "", history_path: "Wiki_LM/historique/20260908-politiciens.md" }))
      .toBe("Wiki_LM/historique/20260908-politiciens.md");
  });
  it("query (regime brief) : chemin absent → brief seul", () => {
    expect(formatWikiResult("query", { brief: "Résumé.", history_path: "" })).toBe("Résumé.");
  });
  it("query (regime brief) : les deux absents → message par défaut", () => {
    expect(formatWikiResult("query", { synthesis: "# X" })).toBe("Réponse wiki vide ou inattendue.");
  });
  it("query (regime brief) : échappe &/</> dans le résumé (Telegram parse_mode HTML)", () => {
    expect(formatWikiResult("query", { brief: "État & <politiciens> > citoyens", history_path: "" }))
      .toBe("État &amp; &lt;politiciens&gt; &gt; citoyens");
  });
  it("query (regime full) : ignore brief/history_path même présents", () => {
    expect(formatWikiResult("query",
      { synthesis: "# X", brief: "Résumé.", history_path: "Wiki_LM/historique/20260908-politiciens.md" }, "full"))
      .toBe("# X");
  });
});

describe("runWikiOp", () => {
  const okExec = (stdout: string) => async () => ({ code: 0, stdout, stderr: "" });

  it("parse le JSON et formate", async () => {
    const out = await runWikiOp(null, "query", "tee gpu",
      okExec('{"synthesis": "# GPU TEE", "references": []}'), "full");
    expect(out).toBe("# GPU TEE");
  });
  it("passe op et arg à l'exec", async () => {
    let seen: string[] = [];
    const exec = async (_api: any, argv: string[]) => { seen = argv; return { code: 0, stdout: '{"files":["a.url"]}', stderr: "" }; };
    await runWikiOp(null, "capture", "#x https://e.com", exec);
    expect(seen).toEqual(["python3", "/wiki-tools/wiki.py", "capture", "#x https://e.com"]);
  });
  it("exit non nul → message d'erreur déterministe", async () => {
    const out = await runWikiOp(null, "status", "", async () => ({ code: 1, stdout: "", stderr: "boom" }));
    expect(out).toBe("Erreur wiki : boom");
  });
  it("stdout non-JSON → message d'erreur déterministe", async () => {
    const out = await runWikiOp(null, "status", "", async () => ({ code: 0, stdout: "pas du json", stderr: "" }));
    expect(out).toContain("Erreur wiki");
  });
  it("ignore les lignes de diagnostic avant le JSON (JSON = dernière ligne)", async () => {
    const stdout = '[query] Embeddings absents — mode BM25 uniquement. Lancez embed.py.\n{"synthesis": "# GPU TEE", "references": []}';
    const out = await runWikiOp(null, "query", "tee gpu", async () => ({ code: 0, stdout, stderr: "" }), "full");
    expect(out).toBe("# GPU TEE");
  });
  it("passe le régime à formatWikiResult (full → synthèse verbatim)", async () => {
    const out = await runWikiOp(null, "query", "tee gpu",
      okExec('{"synthesis": "# GPU TEE", "brief": "Résumé.", "history_path": "Wiki_LM/historique/x.md"}'),
      "full");
    expect(out).toBe("# GPU TEE");
  });
  it("régime par défaut (brief) : résumé + chemin, pas la synthèse complète", async () => {
    const out = await runWikiOp(null, "query", "tee gpu",
      okExec('{"synthesis": "# GPU TEE", "brief": "Résumé.", "history_path": "Wiki_LM/historique/x.md"}'));
    expect(out).toBe("Résumé.\n\nWiki_LM/historique/x.md");
  });
});

describe("fetchWikiOpJson", () => {
  it("succès : ok=true, texte formaté", async () => {
    const out = await fetchWikiOpJson(null, "delete_preview", "src-a",
      async () => ({ code: 0, stdout: '{"status":"ok","slug":"src-a","affected":["src-a"]}', stderr: "" }));
    expect(out).toEqual({ ok: true, text: "1 page(s) seraient affectées :\nsrc-a" });
  });
  it("erreur métier (json.error) : ok=false", async () => {
    const out = await fetchWikiOpJson(null, "delete_preview", "src-x",
      async () => ({ code: 0, stdout: '{"error":"Page introuvable pour le slug \'src-x\'"}', stderr: "" }));
    expect(out).toEqual({ ok: false, text: "Page introuvable pour le slug 'src-x'" });
  });
  it("exit non nul : ok=false", async () => {
    const out = await fetchWikiOpJson(null, "delete_preview", "src-a",
      async () => ({ code: 1, stdout: "", stderr: "boom" }));
    expect(out).toEqual({ ok: false, text: "Erreur wiki : boom" });
  });
  it("stdout non-JSON : ok=false", async () => {
    const out = await fetchWikiOpJson(null, "delete_preview", "src-a",
      async () => ({ code: 0, stdout: "pas du json", stderr: "" }));
    expect(out.ok).toBe(false);
    expect(out.text).toContain("Erreur wiki");
  });
});
