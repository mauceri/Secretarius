// Dépôt des pièces jointes reçues sur Session dans le dossier Réception du coffre.
import { copyFileSync, existsSync, mkdirSync, renameSync, unlinkSync, writeFileSync } from "node:fs";
import { basename, extname, join } from "node:path";
const deux = (n) => String(n).padStart(2, "0");
export const horodatage = (d) => `${d.getFullYear()}-${deux(d.getMonth() + 1)}-${deux(d.getDate())} ${deux(d.getHours())}${deux(d.getMinutes())}`;
export function assainir(nom) {
    let n = basename(nom.replace(/\\/g, "/")).replace(/[/\\:\0\x00-\x1f\x7f]/g, "").trim();
    if (!n || n === "." || n === "..")
        n = "piece-jointe";
    // Limite en OCTETS (NAME_MAX = 255) : le nom final porte aussi l'horodatage et un suffixe éventuel.
    if (Buffer.byteLength(n) > 180) {
        const ext = extname(n).slice(0, 12);
        let racine = "";
        for (const c of n.slice(0, n.length - ext.length)) {
            if (Buffer.byteLength(racine + c) + Buffer.byteLength(ext) > 180)
                break;
            racine += c;
        }
        n = racine + ext;
    }
    return n;
}
function libre(dir, nom) {
    if (!existsSync(join(dir, nom)))
        return nom;
    const ext = extname(nom);
    const racine = ext ? nom.slice(0, -ext.length) : nom;
    for (let i = 2;; i++) {
        const essai = `${racine} (${i})${ext}`;
        if (!existsSync(join(dir, essai)))
            return essai;
    }
}
function deplacer(src, dest) {
    try {
        renameSync(src, dest);
    }
    catch {
        copyFileSync(src, dest);
        unlinkSync(src);
    }
}
export function deposerPiecesJointes(evt, dir, now) {
    mkdirSync(dir, { recursive: true });
    const h = horodatage(now);
    const fichiers = [];
    const lignes = [];
    for (const att of evt.attachments) {
        if (att.error || !att.path) {
            lignes.push(`- ${assainir(att.name)} (${att.error === "echec-telechargement" ? "téléchargement impossible" : "trop grande, non reçue"})`);
            continue;
        }
        const nom = libre(dir, `${h} ${assainir(att.name)}`);
        const dest = join(dir, nom);
        deplacer(att.path, dest);
        fichiers.push(dest);
        lignes.push(`![[${nom}]]`);
    }
    const note = join(dir, libre(dir, `${h} Session.md`));
    const contenu = [
        "---",
        "origine: session",
        `date: ${now.toISOString()}`,
        `expediteur: ${evt.from.slice(0, 8)}`,
        "---",
        "",
        evt.text,
        "",
        ...lignes,
        "",
    ].join("\n");
    writeFileSync(note, contenu);
    return { note, fichiers };
}
