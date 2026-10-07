// gog exécuté directement dans son conteneur, sans sous-agent (2026-10-07) :
// un sous-agent lancé depuis le hook before_agent_reply attend une place sur la
// voie « main » que le tour en cours occupe (« lane wait exceeded », run timeout).
// Même mécanisme que les opérations du wiki (execSandbox).
import { execSandbox } from "./wiki-ops.js";

const GOG_SANDBOX_SESSION_KEY = "agent:gog:subagent:ops";

type Exec = (api: any, argv: string[], script?: string) =>
  Promise<{ code: number; stdout: string; stderr: string }>;

export const execGogSandbox: Exec = (api, argv, script) =>
  execSandbox(api, GOG_SANDBOX_SESSION_KEY, argv, script, "gog");

function erreurGog(stderr: string): string {
  if (/invalid_grant|unauthori[sz]ed|no (refresh )?token|not authenticated/i.test(stderr)) {
    return "Autorisation Google expirée ou absente : lancez /connecter.";
  }
  return `Erreur gog : ${(stderr.trim().split("\n")[0] || "échec inconnu").slice(0, 300)}`;
}

export function formaterDrive(stdout: string): string {
  let data: any;
  try {
    data = JSON.parse(stdout);
  } catch {
    return `Réponse de gog illisible : ${stdout.slice(0, 200)}`;
  }
  const files: any[] = Array.isArray(data) ? data : (data?.files ?? []);
  if (files.length === 0) return "Aucun fichier trouvé.";
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

export async function gogDriveSearch(api: any, q: string, exec: Exec = execGogSandbox): Promise<string> {
  if (!q.trim()) return "Usage : /drive <critères>";
  const r = await exec(api, ["gog", "drive", "search", q.trim(), "--max", "10", "--json"]);
  return r.code === 0 ? formaterDrive(r.stdout) : erreurGog(r.stderr);
}

// Lance le pont OAuth (gog-auth-bridge) en arrière-plan dans le conteneur : il
// expose l'URL dans /gog-config/auth_url puis attend auth_response (fichiers
// partagés avec l'hôte, lus par le plugin).
export async function gogAuthStart(api: any, email: string, exec: Exec = execGogSandbox) {
  return exec(api, ["gog-auth-bridge", email], 'nohup "$1" "$2" >/dev/null 2>&1 &');
}
