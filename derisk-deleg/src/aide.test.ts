import { describe, expect, it } from "vitest";
import { AIDE } from "./aide.js";
import { COMMANDES_TAPEES } from "./dispatch.js";

describe("/aide", () => {
  it("cite toutes les commandes Telegram, routées ou directes", () => {
    const directes = ["/confirm", "/annuler", "/maj", "/jeton", "/aide"];
    const absentes = [...COMMANDES_TAPEES, ...directes].filter(
      (c) => !new RegExp(`^${c.replace("/", "\\/")}\\b`, "m").test(AIDE),
    );
    expect(absentes).toEqual([]);
  });

  it("tient dans un message Telegram", () => {
    expect(AIDE.length).toBeLessThan(4000);
  });
});
