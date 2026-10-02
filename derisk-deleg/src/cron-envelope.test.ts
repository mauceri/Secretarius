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
});
