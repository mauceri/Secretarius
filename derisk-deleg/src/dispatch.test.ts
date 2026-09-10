import { describe, expect, it } from "vitest";
import { commandToAction } from "./dispatch.js";

describe("commandToAction", () => {
  it("mappe /source vers scout, pas wiki", () => {
    expect(commandToAction("/source")).toEqual({ kind: "scout" });
  });

  it("mappe /repondre vers gog-reply (jamais gog direct)", () => {
    expect(commandToAction("/repondre")).toEqual({ kind: "gog-reply" });
  });

  it("mappe /chercher vers gog search", () => {
    expect(commandToAction("/chercher")).toEqual({ kind: "gog", op: "search" });
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

  it("mappe /lire vers gog get", () => {
    expect(commandToAction("/lire")).toEqual({ kind: "gog", op: "get" });
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
});
