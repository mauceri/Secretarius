import { describe, expect, it } from "vitest";
import { formatWikiResult } from "./wiki-ops.js";

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
  it("kb_update : succès et erreur", () => {
    expect(formatWikiResult("kb_update", { status: "done" })).toBe("Base de connaissances mise à jour.");
    expect(formatWikiResult("kb_update", { status: "error", reason: "clusterings/ introuvable" }))
      .toBe("clusterings/ introuvable");
  });
  it("erreur générique inconnue → message par défaut", () => {
    expect(formatWikiResult("query", {})).toBe("Réponse wiki vide ou inattendue.");
  });
});
