// Réception : boucle SSE vers le broker, aiguillage des événements.
import type { InboundEvent } from "./broker-client.js";
import type { deposerPiecesJointes } from "./depot.js";

export type TraiterDeps = {
  allowFrom: string[];
  seen: Set<string>;
  receptionDir: string;
  depot: typeof deposerPiecesJointes;
  dispatchTexte(evt: InboundEvent): Promise<void>;
  repondre(to: string, text: string): Promise<void>;
  ack(id: string): Promise<void>;
};

export async function traiterEvenement(evt: InboundEvent, d: TraiterDeps): Promise<"ignore" | "doublon" | "depot" | "texte"> {
  // Défense en profondeur : le broker filtre déjà, le plugin refiltre.
  if (!d.allowFrom.includes(evt.from)) { await d.ack(evt.id); return "ignore"; }
  if (d.seen.has(evt.id)) { await d.ack(evt.id); return "doublon"; }
  if (evt.attachments.length > 0) {
    const { fichiers } = d.depot(evt, d.receptionDir, new Date());
    const tropGrandes = evt.attachments.filter((a) => a.error === "trop-grande").length;
    let msg = `Reçu : ${fichiers.length} pièce(s) jointe(s) déposée(s) dans Réception/`;
    const echecs = evt.attachments.filter((a) => a.error === "echec-telechargement").length;
    if (tropGrandes) msg += ` ; ${tropGrandes} pièce(s) trop grande(s), non reçue(s)`;
    if (echecs) msg += ` ; ${echecs} pièce(s) non téléchargeable(s)`;
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

export type BoucleDeps = {
  abortSignal: AbortSignal;
  events(signal: AbortSignal): AsyncIterable<InboundEvent>;
  traiter(evt: InboundEvent): Promise<void>;
  sleep(ms: number, signal: AbortSignal): Promise<void>;
  log(line: string): void;
};

export async function boucleReception(d: BoucleDeps): Promise<void> {
  let delai = 1000;
  while (!d.abortSignal.aborted) {
    try {
      for await (const evt of d.events(d.abortSignal)) {
        delai = 1000;
        try { await d.traiter(evt); }
        catch (e) { d.log(`session : échec de traitement de ${evt.id} : ${e instanceof Error ? e.message : String(e)}`); }
      }
    } catch (e) {
      if (d.abortSignal.aborted) return;
      d.log(`session : broker injoignable (${e instanceof Error ? e.message : String(e)}), nouvel essai dans ${delai / 1000} s`);
    }
    if (d.abortSignal.aborted) return;
    await d.sleep(delai, d.abortSignal).catch(() => {});
    delai = Math.min(delai * 2, 60000);
  }
}
