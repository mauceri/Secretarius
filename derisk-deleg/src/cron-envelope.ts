// derisk-deleg/src/cron-envelope.ts
// Enveloppe d'un tour planifié OpenClaw (constatée le 2026-10-02) :
// "[cron:<id> <nom>] <message>\nCurrent time: …préambule…".
// CRON_COMMANDS reflète COMMANDES_PLANIFIABLES de tiron_cron/syntax.py.
export const CRON_COMMANDS: ReadonlySet<string> = new Set([
  "/wikistatus", "/tags", "/relire", "/q", "/r", "/ingest", "/kbupdate",
]);

const ENVELOPE = /^\[cron:[^\s\]]+ ([^\]]+)\] ([\s\S]*?)(?:\nCurrent time:[\s\S]*)?$/;

export function unwrapCronEnvelope(text: string): { jobName: string; message: string } | null {
  const m = ENVELOPE.exec(text);
  return m ? { jobName: m[1], message: m[2].trim() } : null;
}

// Avis qu'OpenClaw place en tête du message suivant après une livraison planifiée
// (« System: [date] A scheduled automation delivered this message… », constaté le
// 2026-10-07) : retirés pour que le message de l'utilisateur soit traité seul.
export function sansAvisSysteme(text: string): string {
  const lignes = text.split("\n");
  let i = 0;
  while (i < lignes.length && lignes[i].startsWith("System: ")) i++;
  if (i === 0) return text;
  while (i < lignes.length && lignes[i].trim() === "") i++;
  return lignes.slice(i).join("\n");
}
