// Correspondance commande routeur -> type d'action. Fonction pure (testable
// sans mock d'api OpenClaw) ; le câblage réel (appel des fonctions delegate*)
// reste dans index.ts, qui est la seule couche à connaître `api`.

export type RouterCommand =
  | "/c" | "/q" | "/ingest" | "/wikistatus" | "/source"
  | "/chercher" | "/connecter" | "/inbox" | "/drive" | "/repondre"
  | "/r" | "/tags" | "/kbupdate" | "/lire" | "/supprimer" | "/relire" | "/verifie"
  | "/cron" | "/ecrire" | "/agenda";

export type ActionKind =
  | { kind: "wiki"; op: "capture" | "query" | "ingest" | "status" | "search" | "tags" | "kb_update" | "delete" | "review" | "verify" }
  | { kind: "scout" }
  | { kind: "gog"; op: "auth_start" | "drive_search" | "agenda" }
  | { kind: "mail"; op: "inbox" | "chercher" | "lire" }
  | { kind: "mail-envoi"; op: "repondre" | "ecrire" }
  | { kind: "cron" };

const TABLE: Record<RouterCommand, ActionKind> = {
  "/c": { kind: "wiki", op: "capture" },
  "/q": { kind: "wiki", op: "query" },
  "/ingest": { kind: "wiki", op: "ingest" },
  "/wikistatus": { kind: "wiki", op: "status" },
  "/source": { kind: "scout" },
  "/chercher": { kind: "mail", op: "chercher" },
  "/connecter": { kind: "gog", op: "auth_start" },
  "/inbox": { kind: "mail", op: "inbox" },
  "/drive": { kind: "gog", op: "drive_search" },
  "/agenda": { kind: "gog", op: "agenda" },
  "/repondre": { kind: "mail-envoi", op: "repondre" },
  "/ecrire": { kind: "mail-envoi", op: "ecrire" },
  "/r": { kind: "wiki", op: "search" },
  "/tags": { kind: "wiki", op: "tags" },
  "/kbupdate": { kind: "wiki", op: "kb_update" },
  "/lire": { kind: "mail", op: "lire" },
  "/supprimer": { kind: "wiki", op: "delete" },
  "/relire": { kind: "wiki", op: "review" },
  "/verifie": { kind: "wiki", op: "verify" },
  "/cron": { kind: "cron" },
};

export const COMMANDES_TAPEES = Object.keys(TABLE);

export function commandToAction(command: string): ActionKind | null {
  return (TABLE as Record<string, ActionKind>)[command] ?? null;
}

// Commandes dont l'argument est obligatoire (même liste que NEEDS_ARG de
// router_service/server.py), avec le texte renvoyé quand il manque.
const USAGES: Record<string, string> = {
  "/c": `/c <texte ou URL>
  texte ou URL : la note à capturer, ou l'adresse d'une page web.
  Options dans le texte :
    #tag      étiquette la capture
    ref:<url> cite une source
    @simple   écrit la note telle quelle comme page source du wiki, sans ingestion
    file:<chemin> ajoute le contenu d'un fichier déjà déposé dans le wiki (pièce jointe)`,
  "/q": `/q <question>
  question : en langage courant. Réponse rédigée à partir du wiki (environ 1 min).`,
  "/r": `/r <mots-clés>
  mots-clés : recherche rapide sans synthèse, 5 résultats.`,
  "/source": `/source <url>
  url : adresse complète de la page (https://…), lue par Scout isolé.`,
  "/chercher": `/chercher <critères> [boîte]
  critères : mots libres, tous exigés. Filtres possibles : de:<expéditeur>,
             objet:<mot>, depuis:<n>j (ex. depuis:7j).
  boîte    : facultative, en dernier mot (gmail, proton, tiron, contact, privacy).
             Sans elle, toutes les boîtes.
  Exemple : /chercher de:paul devis depuis:30j gmail`,
  "/lire": `/lire <id>
  id : identifiant du message, lettre de la boîte et numéro (g:42).
       g gmail, p proton, t tiron, c contact, v privacy.
       Les identifiants sont donnés par /inbox et /chercher.`,
  "/repondre": `/repondre <id> <texte>
  id    : message auquel répondre (g:42), donné par /inbox ou /chercher.
  texte : la réponse, telle qu'elle partira. Aperçu d'abord, puis /confirm.`,
  "/ecrire": `/ecrire <adresse> <texte>
  adresse : adresse électronique complète du destinataire.
  texte   : le message ; sa première ligne sert d'objet. Aperçu d'abord, puis /confirm.`,
  "/supprimer": `/supprimer <page>
  page : nom de fichier de la page sans .md (ex. c-domestication-des-chiens).
         Aperçu des pages touchées en cascade, puis /confirm.`,
  "/verifie": `/verifie <page>
  page : nom de fichier de la page sans .md. Aperçu, puis /confirm.`,
  "/cron": `/cron add <quand> [<canal>] <type> <contenu>
  quand   : "0 8 * * 1" (format cron, entre guillemets), 2026-11-05T09:00, ou 20m / 2h / 1d.
  canal   : facultatif — telegram, whatsapp, sms, obsidian.
  type    : rappel (texte envoyé), commande (/q, /r, /tags…), consigne (tâche pour l'agent).
  contenu : le texte, la commande ou la consigne.
/cron list | rm <id> | pause <id> | reprendre <id>`,
};

export type CommandeTapee =
  | { status: "ok"; command: RouterCommand; args: string }
  | { status: "usage"; reply: string };

// Commande tapée explicitement : reconnue sans routeur (spec 2026-10-12).
// \s couvre l'espace insécable et la tabulation, comme split(None, 1) côté Python.
export function commandeTapee(text: string): CommandeTapee | null {
  const m = text.trim().match(/^(\/\S+)(?:\s+([\s\S]*))?$/);
  if (!m || commandToAction(m[1]) === null) return null;
  const args = m[2] ?? "";
  if (m[1] in USAGES && !args.trim()) return { status: "usage", reply: USAGES[m[1]] };
  return { status: "ok", command: m[1] as RouterCommand, args };
}
