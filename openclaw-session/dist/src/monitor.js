export async function traiterEvenement(evt, d) {
    // Défense en profondeur : le broker filtre déjà, le plugin refiltre.
    if (!d.allowFrom.includes(evt.from)) {
        await d.ack(evt.id);
        return "ignore";
    }
    if (d.seen.has(evt.id)) {
        await d.ack(evt.id);
        return "doublon";
    }
    if (evt.attachments.length > 0) {
        const { fichiers } = d.depot(evt, d.receptionDir, new Date());
        const tropGrandes = evt.attachments.filter((a) => a.error === "trop-grande").length;
        let msg = `Reçu : ${fichiers.length} pièce(s) jointe(s) déposée(s) dans Réception/`;
        if (tropGrandes)
            msg += ` ; ${tropGrandes} pièce(s) trop grande(s), non reçue(s)`;
        d.seen.add(evt.id);
        await d.repondre(evt.from, msg);
        await d.ack(evt.id);
        return "depot";
    }
    await d.dispatchTexte(evt);
    d.seen.add(evt.id);
    await d.ack(evt.id);
    return "texte";
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
