import { describe, expect, it } from "vitest";
import { formatWikiResult, runWikiOp } from "./wiki-ops.js";

describe("formatWikiResult", () => {
  it("query : renvoie la synthèse verbatim", () => {
    expect(formatWikiResult("query", { synthesis: "# GPU TEE\n…", references: ["c-x"] }))
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
  it("kb_update : succès (status ok), async (launched) et erreur", () => {
    expect(formatWikiResult("kb_update", { status: "ok", clustering: "c1" }))
      .toBe("Base de connaissances mise à jour.");
    expect(formatWikiResult("kb_update", { status: "launched" }))
      .toBe("Mise à jour de la base lancée en arrière-plan.");
    expect(formatWikiResult("kb_update", { status: "error", reason: "clusterings/ introuvable" }))
      .toBe("clusterings/ introuvable");
  });
  it("erreur vide → ne renvoie pas un message vide (retombe sur l'op)", () => {
    expect(formatWikiResult("query", { error: "", synthesis: "# X" })).toBe("# X");
  });
  it("erreur générique inconnue → message par défaut", () => {
    expect(formatWikiResult("query", {})).toBe("Réponse wiki vide ou inattendue.");
  });
});

describe("runWikiOp", () => {
  const okExec = (stdout: string) => async () => ({ code: 0, stdout, stderr: "" });

  it("parse le JSON et formate", async () => {
    const out = await runWikiOp(null, "query", "tee gpu",
      okExec('{"synthesis": "# GPU TEE", "references": []}'));
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
    const out = await runWikiOp(null, "query", "tee gpu", async () => ({ code: 0, stdout, stderr: "" }));
    expect(out).toBe("# GPU TEE");
  });
});
