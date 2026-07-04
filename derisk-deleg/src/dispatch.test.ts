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
});
