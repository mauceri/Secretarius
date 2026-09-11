import { describe, expect, it } from "vitest";
import {
  applyWikiResults,
  extractWikiBlocks,
  formatWikiResult,
  SUPPORTED_COMMANDS,
} from "./run-commands";

describe("extractWikiBlocks", () => {
  it("extracts command and argument from a single block", () => {
    const note = "```wiki\n/q\nQuestion ?\n```\n";
    const blocks = extractWikiBlocks(note);
    expect(blocks).toHaveLength(1);
    expect(blocks[0].command).toBe("/q");
    expect(blocks[0].arg).toBe("Question ?");
  });

  it("preserves a multi-line argument joined by \\n, not flattened", () => {
    const note = "```wiki\n/c\nligne 1\nligne 2\nligne 3\n```\n";
    const blocks = extractWikiBlocks(note);
    expect(blocks[0].arg).toBe("ligne 1\nligne 2\nligne 3");
  });

  it("returns an empty argument for a block with no second line", () => {
    const note = "```wiki\n/wikistatus\n```\n";
    const blocks = extractWikiBlocks(note);
    expect(blocks[0].command).toBe("/wikistatus");
    expect(blocks[0].arg).toBe("");
  });

  it("extracts multiple blocks in document order", () => {
    const note = [
      "```wiki",
      "/c",
      "#zoologue https://en.wikipedia.org/wiki/Dmitry_Belyayev_(zoologist)",
      "```",
      "",
      "```wiki",
      "/q",
      "Qu'est-ce que la domestication du renard argenté ?",
      "```",
      "",
    ].join("\n");
    const blocks = extractWikiBlocks(note);
    expect(blocks).toHaveLength(2);
    expect(blocks[0].command).toBe("/c");
    expect(blocks[0].arg).toBe(
      "#zoologue https://en.wikipedia.org/wiki/Dmitry_Belyayev_(zoologist)"
    );
    expect(blocks[1].command).toBe("/q");
    expect(blocks[1].arg).toBe("Qu'est-ce que la domestication du renard argenté ?");
  });

  it("ignores prose text outside ```wiki blocks", () => {
    const note = "Des notes avant.\n\n```wiki\n/tags\n```\n\nDes notes après.\n";
    const blocks = extractWikiBlocks(note);
    expect(blocks).toHaveLength(1);
    expect(blocks[0].command).toBe("/tags");
  });

  it("includes a pre-existing result marker in the block span, for rerun replacement", () => {
    const note =
      "```wiki\n/tags\n```\n\n<!-- wikilm-run:start -->\nancien résultat\n<!-- wikilm-run:end -->\n\nSuite du texte.";
    const blocks = extractWikiBlocks(note);
    expect(blocks).toHaveLength(1);
    const tail = note.slice(blocks[0].end);
    expect(tail.startsWith("\n\nSuite du texte.")).toBe(true);
  });
});

describe("formatWikiResult", () => {
  it("formats an error uniformly regardless of command", () => {
    expect(formatWikiResult("/q", { error: "KB vide" }, true)).toBe("**Erreur :** KB vide");
    expect(formatWikiResult("/q", {}, false)).toBe("**Erreur :** Erreur inconnue");
  });

  it("formats /c with files captured", () => {
    expect(formatWikiResult("/c", { files: ["src-1.md", "src-2.md"] }, true)).toBe(
      "Capturé : src-1.md, src-2.md"
    );
    expect(formatWikiResult("/c", { files: [] }, true)).toBe("Rien à capturer.");
  });

  it("formats /q with the full synthesis", () => {
    expect(formatWikiResult("/q", { synthesis: "Réponse complète." }, true)).toBe(
      "Réponse complète."
    );
  });

  it("formats /ingest per status", () => {
    expect(formatWikiResult("/ingest", { status: "launched", queued: 3 }, true)).toBe(
      "Ingestion lancée (3 en attente)."
    );
    expect(formatWikiResult("/ingest", { status: "already_running", queued: 1 }, true)).toBe(
      "Ingestion déjà en cours."
    );
    expect(formatWikiResult("/ingest", { status: "nothing_to_do", queued: 0 }, true)).toBe(
      "Rien à ingérer."
    );
  });

  it("formats /wikistatus", () => {
    const text = formatWikiResult(
      "/wikistatus",
      { running: false, pending: 2, blocked_files: ["a.url.error"] },
      true
    );
    expect(text).toBe("En cours : non. En attente : 2. Bloqués : 1.");
  });

  it("formats /r as a numbered list", () => {
    const text = formatWikiResult(
      "/r",
      { results: [{ title: "Titre A", excerpt: "Extrait A" }, { title: "Titre B", excerpt: "Extrait B" }] },
      true
    );
    expect(text).toBe("1. Titre A — Extrait A\n2. Titre B — Extrait B");
    expect(formatWikiResult("/r", { results: [] }, true)).toBe("Aucun résultat.");
  });

  it("formats /tags as a comma-separated list", () => {
    expect(formatWikiResult("/tags", { tags: ["ia", "wiki"] }, true)).toBe("ia, wiki");
    expect(formatWikiResult("/tags", { tags: [] }, true)).toBe("Aucun tag.");
  });

  it("formats /kbupdate", () => {
    expect(formatWikiResult("/kbupdate", { status: "ok", clustering: "2026-09-11" }, true)).toBe(
      "Base de connaissances mise à jour (clustering 2026-09-11)."
    );
  });

  it("formats /kbupdate failure reported via status/reason, not error", () => {
    const text = formatWikiResult(
      "/kbupdate",
      { status: "error", reason: "aucun clustering disponible" },
      true
    );
    expect(text).toBe("**Erreur :** aucun clustering disponible");
    expect(text).not.toContain("undefined");
  });

  it("formats /relire empty or with content", () => {
    expect(formatWikiResult("/relire", { status: "empty" }, true)).toBe("Rien à relire.");
    expect(formatWikiResult("/relire", { status: "ok", slug: "src-x", content: "# Contenu" }, true)).toBe(
      "# Contenu"
    );
  });

  it("formats /verifie", () => {
    expect(formatWikiResult("/verifie", { status: "ok", slug: "src-x" }, true)).toBe(
      "Page src-x marquée vérifiée."
    );
  });

  it("formats an unsupported command as a client-side refusal", () => {
    expect(formatWikiResult("/supprimer", {}, true)).toBe(
      "Commande non prise en charge en exécution par lot : /supprimer"
    );
  });
});

describe("SUPPORTED_COMMANDS", () => {
  it("excludes /supprimer", () => {
    expect(SUPPORTED_COMMANDS).not.toContain("/supprimer");
  });

  it("contains exactly the nine wiki commands from the spec", () => {
    expect([...SUPPORTED_COMMANDS].sort()).toEqual(
      ["/c", "/ingest", "/kbupdate", "/q", "/r", "/relire", "/tags", "/verifie", "/wikistatus"].sort()
    );
  });
});

describe("applyWikiResults", () => {
  it("inserts the result right after the block, preserving surrounding prose", () => {
    const note = "Avant.\n\n```wiki\n/tags\n```\n\nAprès.";
    const blocks = extractWikiBlocks(note);
    const output = applyWikiResults(note, blocks, ["ia, wiki"]);
    expect(output).toBe(
      "Avant.\n\n```wiki\n/tags\n```\n\n<!-- wikilm-run:start -->\nia, wiki\n<!-- wikilm-run:end -->\n\nAprès."
    );
  });

  it("replaces a prior result instead of duplicating it on rerun", () => {
    const note = "```wiki\n/tags\n```\n";
    const firstPass = applyWikiResults(note, extractWikiBlocks(note), ["ancien résultat"]);
    const secondBlocks = extractWikiBlocks(firstPass);
    expect(secondBlocks).toHaveLength(1);

    const secondPass = applyWikiResults(firstPass, secondBlocks, ["nouveau résultat"]);
    const marks = secondPass.match(/wikilm-run:start/g) ?? [];
    expect(marks).toHaveLength(1);
    expect(secondPass).toContain("nouveau résultat");
    expect(secondPass).not.toContain("ancien résultat");
  });
});
