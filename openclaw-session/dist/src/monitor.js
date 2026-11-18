// Réception : boucle SSE vers le broker, aiguillage des événements.
import { mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
// États des messages persistés sur disque : un redémarrage du gateway ne doit
// ni relancer un /c déjà parti, ni rejouer un message déjà traité.
export class Vus {
    fichier;
    max;
    etats = {};
    constructor(fichier, max = 2000) {
        this.fichier = fichier;
        this.max = max;
        try {
            this.etats = JSON.parse(readFileSync(fichier, "utf8"));
        }
        catch {
            this.etats = {};
        }
    }
    etat(id) { return this.etats[id]; }
    marquer(id, etat) {
        delete this.etats[id];
        this.etats[id] = etat;
        const cles = Object.keys(this.etats);
        for (const k of cles.slice(0, Math.max(0, cles.length - this.max)))
            delete this.etats[k];
        mkdirSync(dirname(this.fichier), { recursive: true, mode: 0o700 });
        const tmp = `${this.fichier}.tmp`;
        writeFileSync(tmp, JSON.stringify(this.etats), { mode: 0o600 });
        renameSync(tmp, this.fichier);
    }
}
export async function traiterEvenement(evt, d) {
    // Défense en profondeur : le broker filtre déjà, le plugin refiltre.
    if (!d.allowFrom.includes(evt.from)) {
        await d.ack(evt.id);
        return "ignore";
    }
    const etat = d.seen.etat(evt.id);
    if (etat === "fait") {
        await d.ack(evt.id);
        return "doublon";
    }
    if (etat === "en-cours") {
        // Laissé « en cours » par un processus précédent : ne pas relancer (effets de bord possibles).
        d.seen.marquer(evt.id, "fait");
        await d.repondre(evt.from, "Message interrompu par un redémarrage de Secretarius. Renvoyez-le si besoin.");
        await d.ack(evt.id);
        return "interrompu";
    }
    d.seen.marquer(evt.id, "en-cours");
    let resultat;
    try {
        if (evt.attachments.length > 0) {
            const { fichiers } = d.depot(evt, d.receptionDir, new Date());
            const tropGrandes = evt.attachments.filter((a) => a.error === "trop-grande").length;
            const echecs = evt.attachments.filter((a) => a.error === "echec-telechargement").length;
            let msg = `Reçu : ${fichiers.length} pièce(s) jointe(s) déposée(s) dans Réception/`;
            if (tropGrandes)
                msg += ` ; ${tropGrandes} pièce(s) trop grande(s), non reçue(s)`;
            if (echecs)
                msg += ` ; ${echecs} pièce(s) non téléchargeable(s)`;
            await d.repondre(evt.from, msg);
            resultat = "depot";
        }
        else {
            await d.dispatchTexte(evt);
            resultat = "texte";
        }
    }
    catch (e) {
        await d.repondre(evt.from, `Message non traité (erreur : ${e instanceof Error ? e.message : String(e)}). Renvoyez-le si besoin.`).catch(() => { });
        resultat = "erreur";
    }
    d.seen.marquer(evt.id, "fait");
    await d.ack(evt.id);
    return resultat;
}
export async function boucleReception(d) {
    let delai = 1000;
    while (!d.abortSignal.aborted) {
        try {
            for await (const evt of d.events(d.abortSignal)) {
                delai = 1000;
                try {
                    await d.traiter(evt);
                }
                catch (e) {
                    d.log(`session : échec de traitement de ${evt.id} : ${e instanceof Error ? e.message : String(e)}`);
                }
            }
        }
        catch (e) {
            if (d.abortSignal.aborted)
                return;
            d.log(`session : broker injoignable (${e instanceof Error ? e.message : String(e)}), nouvel essai dans ${delai / 1000} s`);
        }
        if (d.abortSignal.aborted)
            return;
        await d.sleep(delai, d.abortSignal).catch(() => { });
        delai = Math.min(delai * 2, 60000);
    }
}
