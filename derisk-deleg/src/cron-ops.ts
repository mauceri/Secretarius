// derisk-deleg/src/cron-ops.ts
// Appel de la CLI JSON tiron_cron (cf. docs/superpowers/specs/2026-10-01-cron-tiron-design.md).
// Asynchrone : le repli qwen3:8b peut prendre des dizaines de secondes, il ne
// doit jamais bloquer la boucle d'événements du gateway.
import { execFile } from "node:child_process";
import { homedir } from "node:os";
import { promisify } from "node:util";

const execFileP = promisify(execFile);

export type CronResult = { ok: boolean; text: string; args?: string };

export async function runTironCron(argv: string[]): Promise<CronResult> {
  const cwd = process.env.SECRETARIUS_DIR || `${homedir()}/Secretarius`;
  try {
    const { stdout } = await execFileP("python3", ["-m", "tiron_cron", ...argv], {
      cwd,
      timeout: 150000,
      encoding: "utf8",
    });
    const data = JSON.parse(String(stdout));
    return { ok: !!data.ok, text: String(data.text ?? ""), ...(data.args ? { args: String(data.args) } : {}) };
  } catch (e: any) {
    return { ok: false, text: `Planificateur indisponible : ${String(e?.message ?? e).slice(0, 300)}` };
  }
}
