// ASSOMPTION SDK (à vérifier live) : resolveSandboxContext n'est PAS exporté
// par "openclaw/plugin-sdk/sandbox" (absent de son .d.ts / .js malgré le brief) ;
// il l'est en revanche par "openclaw/plugin-sdk/agent-harness-runtime"
// (confirmé dans node_modules/openclaw/dist/plugin-sdk/agent-harness-runtime.d.ts
// et présent dans le "exports" map de openclaw/package.json).
import { resolveSandboxContext } from "openclaw/plugin-sdk/agent-harness-runtime";

// Session stable -> le backend sandbox réutilise le même conteneur entre appels.
const WIKI_SANDBOX_SESSION_KEY = "agent:wiki:subagent:ops";
const WIKI_SANDBOX_TIMEOUT_MS = 120000;

// Exécute argv (ex. ["python3","/wiki-tools/wiki.py","status"]) dans le sandbox
// wiki via le SDK OpenClaw. Ne lève jamais : sandbox indisponible -> code 1.
//
// SandboxBackendCommandParams n'accepte pas d'argv direct : { script: string,
// args?: string[] }. `script` est passé à `sh -c` et `args` devient $1, $2, ...
// (cf. runDockerSandboxShellCommand : docker exec -i <container> sh -c
// <script> openclaw-sandbox-fs <args...>). On construit donc script comme une
// suite de références positionnelles ("$1" "$2" ...) et on passe argv tel
// quel en args, ce qui exécute argv[0] avec argv[1..] comme arguments, sans
// interpolation de texte dans le script (pas d'injection shell).
export async function execWikiSandbox(
  api: any,
  argv: string[],
): Promise<{ code: number; stdout: string; stderr: string }> {
  try {
    const config = api?.config ?? api?.runtime?.config?.current?.();
    const ctx = await resolveSandboxContext({
      config,
      sessionKey: WIKI_SANDBOX_SESSION_KEY,
    });
    if (!ctx || !ctx.enabled || !ctx.backend) {
      return { code: 1, stdout: "", stderr: "sandbox wiki indisponible" };
    }

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), WIKI_SANDBOX_TIMEOUT_MS);
    try {
      const script = argv.map((_, i) => `"$${i + 1}"`).join(" ");
      const res = await ctx.backend.runShellCommand({
        script,
        args: argv,
        allowFailure: true,
        signal: controller.signal,
      });
      return {
        code: res.code,
        stdout: res.stdout?.toString("utf8") ?? "",
        stderr: res.stderr?.toString("utf8") ?? "",
      };
    } finally {
      clearTimeout(timer);
    }
  } catch (err) {
    return {
      code: 1,
      stdout: "",
      stderr: err instanceof Error ? err.message : String(err),
    };
  }
}

type Exec = (api: any, argv: string[]) => Promise<{ code: number; stdout: string; stderr: string }>;

// "full" = synthèse complète (Obsidian, WebChat — rendent le Markdown
// correctement). "brief" = résumé court + lien vers l'historique (Telegram,
// et tout canal non identifiable). Voir la spec pour le détail par canal.
export type WikiOpRegime = "brief" | "full";

// Compose execWikiSandbox : construit argv, exécute, parse JSON, formate ou renvoie erreur.
export async function runWikiOp(
  api: any, op: string, arg: string, exec: Exec = execWikiSandbox,
  regime: WikiOpRegime = "brief",
): Promise<string> {
  const argv = ["python3", "/wiki-tools/wiki.py", op];
  if (arg) argv.push(arg);
  const { code, stdout, stderr } = await exec(api, argv);
  if (code !== 0) return `Erreur wiki : ${(stderr || stdout || "échec").slice(0, 500)}`;
  // wiki.py imprime parfois des lignes de diagnostic avant le JSON (ex. query :
  // « [query] Embeddings absents… ») ; le résultat JSON est toujours la DERNIÈRE
  // ligne non vide (un seul print(json.dumps(...)) final, sur une ligne).
  const lines = stdout.split("\n").map((l) => l.trim()).filter(Boolean);
  const lastLine = lines.length ? lines[lines.length - 1] : "";
  let json: any;
  try {
    json = JSON.parse(lastLine);
  } catch {
    return `Erreur wiki : sortie inattendue (${stdout.slice(0, 200)})`;
  }
  return formatWikiResult(op, json, regime);
}

// Le canal Telegram envoie toujours en parse_mode "HTML" (jamais Markdown) :
// un obsidian:// nu n'y est jamais cliquable (Telegram ne détecte pas les
// schémas d'URI personnalisés en mode HTML), et tout &/</> non échappé dans
// un texte généré par LLM casserait le rendu du message.
function escapeHtml(text: string): string {
  return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// Formatage déterministe du JSON de wiki.py en message utilisateur.
// Aucune invention : sur erreur, on surface le texte de wiki.py verbatim.
export function formatWikiResult(op: string, json: any, regime: WikiOpRegime = "brief"): string {
  if (json && typeof json.error === "string" && json.error.trim()) return json.error;
  if (json && json.status === "error") return json.reason ?? json.error ?? "Erreur wiki.";

  switch (op) {
    case "query": {
      if (regime === "full") {
        return typeof json?.synthesis === "string" && json.synthesis.trim()
          ? json.synthesis
          : "Réponse wiki vide ou inattendue.";
      }
      const brief = typeof json?.brief === "string" ? json.brief.trim() : "";
      const uri = typeof json?.obsidian_uri === "string" ? json.obsidian_uri : "";
      if (!brief && !uri) return "Réponse wiki vide ou inattendue.";
      // PAS de <a href="obsidian://...">: envoyer ce schéma d'URI comme
      // lien cliquable a bloqué la livraison Telegram en test réel (la
      // génération de la réponse est rapide — 6s — mais rien n'était jamais
      // envoyé, probablement une tentative d'aperçu de lien qui reste
      // bloquée sur un schéma non http(s)). Texte brut échappé : fiable
      // mais non cliquable, en attendant une solution qui l'est aussi.
      const parts: string[] = [];
      if (brief) parts.push(escapeHtml(brief));
      if (uri) parts.push(escapeHtml(uri));
      return parts.join("\n\n");
    }
    case "capture": {
      const files = Array.isArray(json?.files) ? json.files : [];
      return files.length
        ? `Capturé : ${files.join(", ")} (en file d'attente pour ingestion).`
        : "Rien à capturer.";
    }
    case "ingest":
      if (json?.status === "launched") return "Ingestion lancée en arrière-plan.";
      if (json?.status === "nothing_to_do") return "Rien à ingérer.";
      if (json?.status === "already_running") return "Ingestion déjà en cours.";
      return "État d'ingestion inconnu.";
    case "status": {
      const state = json?.running ? "Ingestion en cours" : "Ingestion à l'arrêt";
      const lr = json?.last_run;
      const lrTxt = json?.running || !lr
        ? ""
        : ` (dernier run : ${lr.ingested ?? "?"} ingéré(s), ${lr.errors ?? "?"} erreur(s))`;
      const blocked = Array.isArray(json?.blocked_files) ? json.blocked_files.length : 0;
      return `${state}${lrTxt}. En attente : ${json?.pending ?? 0}. Bloqués : ${blocked}.`;
    }
    case "tags": {
      const tags = Array.isArray(json?.tags) ? json.tags : [];
      return tags.length ? `Tags : ${tags.join(", ")}.` : "Aucun tag.";
    }
    case "search": {
      const results = Array.isArray(json?.results) ? json.results : [];
      if (!results.length) return "Aucun résultat.";
      return results
        .map((r: any, i: number) => `${i + 1}. ${r.title}\n   ${r.excerpt}`)
        .join("\n\n");
    }
    case "kb_update":
      // op_kb_update est synchrone (status "ok" = mise à jour faite) ; on gère
      // aussi "launched" au cas où on brancherait le worker async, sans sur-claim.
      if (json?.status === "launched") return "Mise à jour de la base lancée en arrière-plan.";
      return "Base de connaissances mise à jour.";
    default:
      return "Réponse wiki vide ou inattendue.";
  }
}
