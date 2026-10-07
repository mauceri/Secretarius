import { describe, expect, it, vi } from "vitest";
import { formaterDrive, gogDriveSearch, gogAuthStart } from "./gog-ops.js";

const JSON_DRIVE = JSON.stringify({ files: [
  { name: "Thèse — chapitre 1", mimeType: "application/vnd.google-apps.document",
    modifiedTime: "2026-09-12T10:00:00.000Z", webViewLink: "https://docs.google.com/d/1" },
  { name: "Bibliographie.pdf", mimeType: "application/pdf",
    modifiedTime: "2026-08-01T08:30:00.000Z", webViewLink: "https://drive.google.com/f/2" },
] });

describe("formaterDrive", () => {
  it("liste nom, date et lien", () => {
    expect(formaterDrive(JSON_DRIVE)).toBe(
      "1. Thèse — chapitre 1 (12/09/2026)\n   https://docs.google.com/d/1\n" +
      "2. Bibliographie.pdf (01/08/2026)\n   https://drive.google.com/f/2");
  });
  it("accepte un tableau nu et signale l'absence de résultat", () => {
    expect(formaterDrive("[]")).toBe("Aucun fichier trouvé.");
  });
  it("sortie illisible", () => {
    expect(formaterDrive("pas du json")).toMatch(/^Réponse de gog illisible/);
  });
});

describe("gogDriveSearch (appel direct, sans sous-agent)", () => {
  it("exécute gog drive search dans le conteneur gog", async () => {
    const exec = vi.fn(async () => ({ code: 0, stdout: JSON_DRIVE, stderr: "" }));
    const out = await gogDriveSearch({}, "thèse", exec);
    expect(exec).toHaveBeenCalledWith({}, ["gog", "drive", "search", "thèse", "--max", "10", "--json"]);
    expect(out).toMatch(/^1\. Thèse/);
  });
  it("sans critère : usage", async () => {
    const exec = vi.fn();
    expect(await gogDriveSearch({}, "  ", exec)).toBe("Usage : /drive <critères>");
    expect(exec).not.toHaveBeenCalled();
  });
  it("autorisation expirée : invite à /connecter", async () => {
    const exec = vi.fn(async () => ({ code: 1, stdout: "", stderr: 'oauth2: "invalid_grant" "Bad Request"' }));
    expect(await gogDriveSearch({}, "thèse", exec)).toBe("Autorisation Google expirée ou absente : lancez /connecter.");
  });
  it("autre erreur : première ligne relayée", async () => {
    const exec = vi.fn(async () => ({ code: 1, stdout: "", stderr: "sandbox gog indisponible\ndétail" }));
    expect(await gogDriveSearch({}, "thèse", exec)).toBe("Erreur gog : sandbox gog indisponible");
  });
});

describe("gogAuthStart", () => {
  it("lance le pont OAuth en arrière-plan dans le conteneur gog", async () => {
    const exec = vi.fn(async () => ({ code: 0, stdout: "", stderr: "" }));
    await gogAuthStart({}, "moi@gmail.com", exec);
    const [, argv, script] = exec.mock.calls[0] as any[];
    expect(argv).toEqual(["gog-auth-bridge", "moi@gmail.com"]);
    expect(script).toBe('nohup "$1" "$2" >/dev/null 2>&1 &');
  });
});
