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

  it("returns the body as-is when shorter than the incipit length", () => {
    const result = buildCaptureText({ body: "Courte note.", title: "T", path: "p.md" });
    expect(result.endsWith("Courte note.")).toBe(true);
    expect(result).not.toContain("…");
  });

  it("truncates long bodies at the last space before 200 characters", () => {
    const body = "mot ".repeat(60).trim();
    const result = buildCaptureText({ body, title: "T", path: "p.md" });
    const content = result.split("\n\n")[1];
    expect(content.endsWith("…")).toBe(true);
    expect(content.length).toBeLessThanOrEqual(201);
    expect(content).not.toMatch(/ …$/);
  });

  it("extracts a level-1 Résumé section verbatim, ignoring length limit", () => {
    const longSummary = "Phrase de résumé assez longue pour dépasser deux cents caractères si on la répète. ".repeat(5).trim();
    const body = `# Résumé\n\n${longSummary}\n\n## Autre section\n\nIgnoré.`;
    const result = buildCaptureText({ body, title: "T", path: "p.md" });
    const content = result.slice(result.indexOf("\n\n") + 2);
    expect(content.trim()).toBe(longSummary);
    expect(content).not.toContain("Ignoré");
  });

  it("matches '# Summary' case-insensitively", () => {
    const body = "# summary\nHello world.";
    const result = buildCaptureText({ body, title: "T", path: "p.md" });
    expect(result.endsWith("Hello world.")).toBe(true);
  });

  it("does not treat a level-2 heading as a summary section", () => {
    const body = "## Résumé\nTexte court.";
    const result = buildCaptureText({ body, title: "T", path: "p.md" });
    expect(result.endsWith("## Résumé\nTexte court.")).toBe(true);
  });
});
