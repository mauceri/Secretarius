import { describe, it, expect, beforeEach } from "vitest";
import { mkdtempSync, writeFileSync, readdirSync, readFileSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { deposerPiecesJointes } from "./depot.js";

const MOI = "05" + "a".repeat(64);
let media: string, recep: string;
beforeEach(() => {
  media = mkdtempSync(join(tmpdir(), "media-"));
  recep = join(mkdtempSync(join(tmpdir(), "arbath-")), "Réception");
});
const now = new Date(2026, 10, 18, 15, 2);
function pj(name: string, contenu = "x") {
  const p = join(media, Math.random().toString(36).slice(2));
  writeFileSync(p, contenu);
  return { path: p, name, contentType: "image/png", size: contenu.length };
}
const evt = (atts: any[], text = "ma légende") => ({ id: "m1", from: MOI, timestamp: 1, text, attachments: atts });

describe("deposerPiecesJointes", () => {
  it("noms hostiles : tout reste dans Réception", () => {
    const r = deposerPiecesJointes(evt([pj("../../.bashrc"), pj("/etc/passwd"), pj(""), pj("a".repeat(300) + ".png")]), recep, now);
    for (const f of r.fichiers) expect(dirname(f)).toBe(recep);
    const noms = readdirSync(recep);
    expect(noms.some((n) => n.includes("/"))).toBe(false);
    expect(noms).toContain("2026-11-18 1502 .bashrc");
    expect(noms).toContain("2026-11-18 1502 passwd");
    expect(noms).toContain("2026-11-18 1502 piece-jointe");
    expect(r.fichiers.every((f) => f.length - recep.length < 140)).toBe(true);
  });
  it("même nom deux fois : suffixe (2), rien d'écrasé", () => {
    deposerPiecesJointes(evt([pj("photo.png", "un")]), recep, now);
    const r = deposerPiecesJointes(evt([pj("photo.png", "deux")]), recep, now);
    expect(readFileSync(join(recep, "2026-11-18 1502 photo.png"), "utf8")).toBe("un");
    expect(r.fichiers[0]).toBe(join(recep, "2026-11-18 1502 photo (2).png"));
    expect(readFileSync(r.fichiers[0], "utf8")).toBe("deux");
  });
  it("note : en-tête, texte et un lien par fichier ; fichier retiré du dossier du broker", () => {
    const a = pj("photo.png");
    const r = deposerPiecesJointes(evt([a]), recep, now);
    const note = readFileSync(r.note, "utf8");
    expect(r.note).toBe(join(recep, "2026-11-18 1502 Session.md"));
    expect(note).toContain("origine: session");
    expect(note).toContain(`expediteur: ${MOI.slice(0, 8)}`);
    expect(note).not.toContain(MOI);
    expect(note).toContain("ma légende");
    expect(note).toContain("![[2026-11-18 1502 photo.png]]");
    expect(existsSync(a.path)).toBe(false);
  });
  it("pièce trop grande : pas de fichier, signalée dans la note", () => {
    const r = deposerPiecesJointes(evt([{ path: "", name: "gros.bin", contentType: "x", size: 99, error: "trop-grande" }]), recep, now);
    expect(r.fichiers).toEqual([]);
    expect(readFileSync(r.note, "utf8")).toContain("gros.bin (trop grande, non reçue)");
  });
});
