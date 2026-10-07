// Appel de la CLI JSON tiron_mail (IMAP/SMTP), sans sous-agent ni gog
// (spec docs/superpowers/specs/2026-10-07-agent-courrier-design.md).
import { runPythonModule, type CronResult } from "./cron-ops.js";

export function runTironMail(argv: string[]): Promise<CronResult> {
  return runPythonModule("tiron_mail", argv, "Courrier indisponible");
}
