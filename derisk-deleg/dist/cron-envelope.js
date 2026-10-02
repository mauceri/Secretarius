// derisk-deleg/src/cron-envelope.ts
// Enveloppe d'un tour planifié OpenClaw (constatée le 2026-10-02) :
// "[cron:<id> <nom>] <message>\nCurrent time: …préambule…".
// CRON_COMMANDS reflète COMMANDES_PLANIFIABLES de tiron_cron/syntax.py.
export const CRON_COMMANDS = new Set([
    "/wikistatus", "/tags", "/relire", "/q", "/r", "/ingest", "/kbupdate",
]);
const ENVELOPE = /^\[cron:[^\s\]]+ ([^\]]+)\] ([\s\S]*?)(?:\nCurrent time:[\s\S]*)?$/;
export function unwrapCronEnvelope(text) {
    const m = ENVELOPE.exec(text);
    return m ? { jobName: m[1], message: m[2].trim() } : null;
}
