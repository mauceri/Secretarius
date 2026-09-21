import { describe, expect, it } from "vitest";
import { buildCaptureText } from "./capture-text";

describe("buildCaptureText", () => {
  it("always prefixes with the origin line", () => {
    const result = buildCaptureText({
      body: "Courte note.",
      title: "Ma note",
      path: "dossier/ma-note.md",
    });
    expect(result.startsWith("Note d'origine : Ma note (dossier/ma-note.md)\n\n")).toBe(true);
  });

  it("returns the body as-is when there is no Résumé heading", () => {
    const result = buildCaptureText({ body: "Courte note.", title: "T", path: "p.md" });
    expect(result.endsWith("Courte note.")).toBe(true);
  });

  it("captures a long body in full, without truncation, when there is no Résumé heading", () => {
    const body = "mot ".repeat(500).trim();
    const result = buildCaptureText({ body, title: "T", path: "p.md" });
    const content = result.slice(result.indexOf("\n\n") + 2);
    expect(content).toBe(body);
    expect(content).not.toContain("…");
  });

  it("captures a note with a Résumé heading in full — the section is not stripped", () => {
    const longSummary = "Phrase de résumé assez longue pour dépasser deux cents caractères si on la répète. ".repeat(5).trim();
    const body = `# Résumé\n\n${longSummary}\n\n## Autre section\n\nPas ignoré.`;
    const result = buildCaptureText({ body, title: "T", path: "p.md" });
    const content = result.slice(result.indexOf("\n\n") + 2);
    expect(content).toBe(body);
    expect(content).toContain("Pas ignoré");
  });
});
