import { describe, it, expect, vi } from "vitest";
import { traiterEvenement, boucleReception } from "./monitor.js";

const MOI = "05" + "a".repeat(64);
const INCONNU = "05" + "b".repeat(64);
const evt = (o: any = {}) => ({ id: "m1", from: MOI, timestamp: 1, text: "salut", attachments: [], ...o });

function deps(o: any = {}) {
  return {
    allowFrom: [MOI],
    seen: new Set<string>(),
    receptionDir: "/r",
    depot: vi.fn(() => ({ note: "/r/n.md", fichiers: ["/r/f"] })),
    dispatchTexte: vi.fn(async () => {}),
    repondre: vi.fn(async () => {}),
    ack: vi.fn(async () => {}),
    ...o,
  };
}

describe("traiterEvenement", () => {
  it("hors liste : ignore, ack, rien d'autre", async () => {
    const d = deps();
    expect(await traiterEvenement(evt({ from: INCONNU }), d)).toBe("ignore");
    expect(d.ack).toHaveBeenCalledWith("m1");
    expect(d.depot).not.toHaveBeenCalled();
    expect(d.dispatchTexte).not.toHaveBeenCalled();
    expect(d.repondre).not.toHaveBeenCalled();
  });
  it("même id rejoué : dispatch une seule fois", async () => {
    const d = deps();
    expect(await traiterEvenement(evt(), d)).toBe("texte");
    expect(await traiterEvenement(evt(), d)).toBe("doublon");
    expect(d.dispatchTexte).toHaveBeenCalledTimes(1);
    expect(d.ack).toHaveBeenCalledTimes(2);
  });
  it("pièce jointe : dépôt, réponse exacte, pas de dispatch", async () => {
    const d = deps();
    const e = evt({ attachments: [{ path: "/m/1", name: "p.png", contentType: "image/png", size: 3 }] });
    expect(await traiterEvenement(e, d)).toBe("depot");
    expect(d.depot).toHaveBeenCalledWith(e, "/r", expect.any(Date));
    expect(d.repondre).toHaveBeenCalledWith(MOI, "Reçu : 1 pièce(s) jointe(s) déposée(s) dans Réception/");
    expect(d.dispatchTexte).not.toHaveBeenCalled();
  });
  it("pièce trop grande : signalée dans la réponse", async () => {
    const d = deps({ depot: vi.fn(() => ({ note: "/r/n.md", fichiers: [] })) });
    await traiterEvenement(evt({ attachments: [{ path: "", name: "g", contentType: "x", size: 99, error: "trop-grande" }] }), d);
    expect(d.repondre).toHaveBeenCalledWith(MOI, "Reçu : 0 pièce(s) jointe(s) déposée(s) dans Réception/ ; 1 pièce(s) trop grande(s), non reçue(s)");
  });
  it("échec du dispatch : pas d'ack, le rejeu est retraité", async () => {
    const d = deps({ dispatchTexte: vi.fn().mockRejectedValueOnce(new Error("panne")).mockResolvedValue(undefined) });
    await expect(traiterEvenement(evt(), d)).rejects.toThrow("panne");
    expect(d.ack).not.toHaveBeenCalled();
    expect(await traiterEvenement(evt(), d)).toBe("texte");
    expect(d.dispatchTexte).toHaveBeenCalledTimes(2);
  });
  it("texte seul : dispatch puis ack", async () => {
    const d = deps();
    expect(await traiterEvenement(evt(), d)).toBe("texte");
    expect(d.dispatchTexte).toHaveBeenCalledTimes(1);
    expect(d.ack).toHaveBeenCalledWith("m1");
  });
});

describe("boucleReception", () => {
  it("deux échecs de connexion puis un événement : traité, délais croissants", async () => {
    const ac = new AbortController();
    let appels = 0;
    const delais: number[] = [];
    const traites: string[] = [];
    await boucleReception({
      abortSignal: ac.signal,
      events: (_s) => {
        appels++;
        if (appels <= 2) return (async function* () { throw new Error("injoignable"); })();
        return (async function* () { yield evt({ id: "e9" }); ac.abort(); })();
      },
      traiter: async (e) => { traites.push(e.id); },
      sleep: async (ms) => { delais.push(ms); },
      log: () => {},
    });
    expect(traites).toEqual(["e9"]);
    expect(delais[0]).toBe(1000);
    expect(delais[1]).toBe(2000);
  });
  it("abort : rend la main", async () => {
    const ac = new AbortController();
    ac.abort();
    await boucleReception({ abortSignal: ac.signal, events: () => (async function* () {})(), traiter: async () => {}, sleep: async () => {}, log: () => {} });
  });
});
