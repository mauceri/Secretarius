// derisk-deleg/src/cron-envelope.test.ts
import { describe, expect, it } from "vitest";
import { CRON_COMMANDS, unwrapCronEnvelope } from "./cron-envelope.js";

const PREAMBULE = "\nCurrent time: Friday, October 2nd, 2026 - 21:40 (Europe/Paris)\nThis is an unattended scheduled run.";

describe("unwrapCronEnvelope", () => {
  it("extrait nom et message d'origine, sans le préambule", () => {
    expect(unwrapCronEnvelope(`[cron:da77e597-8482 tiron-abcd1234] /wikistatus${PREAMBULE}`))
      .toEqual({ jobName: "tiron-abcd1234", message: "/wikistatus" });
  });
  it("message multiligne conservé", () => {
    expect(unwrapCronEnvelope(`[cron:1 tiron-x] résume\nle wiki${PREAMBULE}`)?.message).toBe("résume\nle wiki");
  });
  it("sans préambule", () => {
    expect(unwrapCronEnvelope("[cron:1 tiron-x] /tags")).toEqual({ jobName: "tiron-x", message: "/tags" });
  });
  it("message ordinaire -> null", () => {
    expect(unwrapCronEnvelope("bonjour [cron:1 x] /tags")).toBeNull();
  });
  it("liste blanche alignée sur tiron_cron", () => {
    expect([...CRON_COMMANDS].sort()).toEqual(["/ingest", "/kbupdate", "/q", "/r", "/relire", "/tags", "/wikistatus"]);
  });

  it("retire les avis système d'OpenClaw placés en tête du message", async () => {
    const { sansAvisSysteme } = await import("./cron-envelope.js");
    const recu = [
      "System: [2026-10-07 10:02:57 GMT+2] A scheduled automation delivered this message to this channel:",
      "System: \"La vie est une série de choix.\" — Sénèque",
      "",
      "/cron add 2m consigne donne-moi une citation de Marc Aurèle",
    ].join("\n");
    expect(sansAvisSysteme(recu)).toBe("/cron add 2m consigne donne-moi une citation de Marc Aurèle");
    expect(sansAvisSysteme("bonjour")).toBe("bonjour");
    expect(sansAvisSysteme("une ligne\nSystem: au milieu")).toBe("une ligne\nSystem: au milieu");
  });
});
