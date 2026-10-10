// gog exécuté directement dans son conteneur, sans sous-agent (2026-10-07) :
// un sous-agent lancé depuis le hook before_agent_reply attend une place sur la
// voie « main » que le tour en cours occupe (« lane wait exceeded », run timeout).
// Même mécanisme que les opérations du wiki (execSandbox).
import { execSandbox } from "./wiki-ops.js";
const GOG_SANDBOX_SESSION_KEY = "agent:gog:subagent:ops";
export const execGogSandbox = (api, argv, script) => execSandbox(api, GOG_SANDBOX_SESSION_KEY, argv, script, "gog");
function erreurGog(stderr) {
    if (/invalid_grant|unauthori[sz]ed|no (refresh )?token|not authenticated/i.test(stderr)) {
        return "Autorisation Google expirée ou absente : lancez /connecter.";
    }
    return `Erreur gog : ${(stderr.trim().split("\n")[0] || "échec inconnu").slice(0, 300)}`;
}
export function formaterDrive(stdout) {
    let data;
    try {
        data = JSON.parse(stdout);
    }
    catch {
        return `Réponse de gog illisible : ${stdout.slice(0, 200)}`;
    }
    const files = Array.isArray(data) ? data : (data?.files ?? []);
    if (files.length === 0)
        return "Aucun fichier trouvé.";
    return files
        .map((f, i) => {
        const d = f.modifiedTime ? new Date(f.modifiedTime) : null;
        const date = d && !isNaN(d.getTime())
            ? ` (${String(d.getUTCDate()).padStart(2, "0")}/${String(d.getUTCMonth() + 1).padStart(2, "0")}/${d.getUTCFullYear()})`
            : "";
        return `${i + 1}. ${f.name}${date}\n   ${f.webViewLink ?? ""}`.trimEnd();
    })
        .join("\n");
}
export async function gogDriveSearch(api, q, exec = execGogSandbox) {
    if (!q.trim())
        return "Usage : /drive <critères>";
    const r = await exec(api, ["gog", "drive", "search", q.trim(), "--max", "10", "--json"]);
    return r.code === 0 ? formaterDrive(r.stdout) : erreurGog(r.stderr);
}
// Date et heure prises telles que gog les donne (déjà dans le fuseau de l'agenda) :
// « 2026-10-12T09:30:00+02:00 » -> « 12/10 09:30 » ; « 2026-10-14 » -> « 14/10 journée ».
function quandEvenement(start) {
    const s = String(start?.dateTime ?? start?.date ?? "");
    const m = s.match(/^\d{4}-(\d{2})-(\d{2})(?:T(\d{2}:\d{2}))?/);
    if (!m)
        return s || "?";
    return `${m[2]}/${m[1]} ${m[3] ?? "journée"}`;
}
export function formaterAgenda(stdout) {
    let data;
    try {
        data = JSON.parse(stdout);
    }
    catch {
        return `Réponse de gog illisible : ${stdout.slice(0, 200)}`;
    }
    const events = Array.isArray(data) ? data : (data?.events ?? data?.items ?? []);
    if (events.length === 0)
        return "Aucun événement.";
    return events
        .map((e) => `${quandEvenement(e.start)}  ${e.summary ?? "(sans titre)"}${e.location ? ` (${e.location})` : ""}`)
        .join("\n");
}
// Lecture seule : événements des n prochains jours (7 par défaut), tous agendas.
export async function gogAgenda(api, args, exec = execGogSandbox) {
    const a = args.trim();
    const jours = a === "" ? 7 : (/^\d+$/.test(a) ? Number(a) : NaN);
    if (!(jours >= 1 && jours <= 60))
        return "Usage : /agenda [jours, de 1 à 60]";
    const r = await exec(api, ["gog", "calendar", "events", "--all", "--days", String(jours), "--max", "30", "--json"]);
    return r.code === 0 ? formaterAgenda(r.stdout) : erreurGog(r.stderr);
}
// Lance le pont OAuth (gog-auth-bridge) en arrière-plan dans le conteneur : il
// expose l'URL dans /gog-config/auth_url puis attend auth_response (fichiers
// partagés avec l'hôte, lus par le plugin).
export async function gogAuthStart(api, email, exec = execGogSandbox) {
    return exec(api, ["gog-auth-bridge", email], 'nohup "$1" "$2" >/dev/null 2>&1 &');
}
