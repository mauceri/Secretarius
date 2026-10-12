import { describe, expect, it } from "vitest";
import { commandToAction, commandeTapee } from "./dispatch.js";

describe("commandToAction", () => {
  it("mappe /source vers scout, pas wiki", () => {
    expect(commandToAction("/source")).toEqual({ kind: "scout" });
  });

  it("mappe /repondre et /ecrire vers l'envoi de courrier (tiron_mail)", () => {
    expect(commandToAction("/repondre")).toEqual({ kind: "mail-envoi", op: "repondre" });
    expect(commandToAction("/ecrire")).toEqual({ kind: "mail-envoi", op: "ecrire" });
  });

  it("mappe /chercher et /inbox vers le courrier (tiron_mail), plus gog", () => {
    expect(commandToAction("/chercher")).toEqual({ kind: "mail", op: "chercher" });
    expect(commandToAction("/inbox")).toEqual({ kind: "mail", op: "inbox" });
  });

  it("retourne null pour une commande inconnue", () => {
    expect(commandToAction("/inexistant")).toBeNull();
  });

  it("mappe /r vers wiki search", () => {
    expect(commandToAction("/r")).toEqual({ kind: "wiki", op: "search" });
  });

  it("mappe /tags vers wiki tags", () => {
    expect(commandToAction("/tags")).toEqual({ kind: "wiki", op: "tags" });
  });

  it("mappe /kbupdate vers wiki kb_update", () => {
    expect(commandToAction("/kbupdate")).toEqual({ kind: "wiki", op: "kb_update" });
  });

  it("mappe /lire vers le courrier ; /drive et /connecter restent gog", () => {
    expect(commandToAction("/lire")).toEqual({ kind: "mail", op: "lire" });
    expect(commandToAction("/drive")).toEqual({ kind: "gog", op: "drive_search" });
    expect(commandToAction("/agenda")).toEqual({ kind: "gog", op: "agenda" });
    expect(commandToAction("/connecter")).toEqual({ kind: "gog", op: "auth_start" });
  });

  it("mappe /supprimer vers wiki delete", () => {
    expect(commandToAction("/supprimer")).toEqual({ kind: "wiki", op: "delete" });
  });

  it("mappe /relire vers wiki review", () => {
    expect(commandToAction("/relire")).toEqual({ kind: "wiki", op: "review" });
  });

  it("mappe /verifie vers wiki verify", () => {
    expect(commandToAction("/verifie")).toEqual({ kind: "wiki", op: "verify" });
  });

  it("mappe /cron vers cron", () => {
    expect(commandToAction("/cron")).toEqual({ kind: "cron" });
  });
});

describe("commandeTapee", () => {
  it("commande connue avec argument", () => {
    expect(commandeTapee("/q quelle heure ?")).toEqual({ status: "ok", command: "/q", args: "quelle heure ?" });
  });
  it("espace insécable ou tabulation après la commande", () => {
    expect(commandeTapee("/q x")).toEqual({ status: "ok", command: "/q", args: "x" });
    expect(commandeTapee("/r\tchiens")).toEqual({ status: "ok", command: "/r", args: "chiens" });
  });
  it("argument requis manquant", () => {
    expect(commandeTapee("/q   ")).toEqual({ status: "usage", reply: "Usage : /q <argument>" });
  });
  it("commande sans argument requis", () => {
    expect(commandeTapee("/wikistatus")).toEqual({ status: "ok", command: "/wikistatus", args: "" });
  });
  it("commande inconnue ou texte libre : null", () => {
    expect(commandeTapee("/inconnue x")).toBeNull();
    expect(commandeTapee("bonjour")).toBeNull();
  });
});
