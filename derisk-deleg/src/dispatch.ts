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

export function commandToAction(command: string): ActionKind | null {
  return (TABLE as Record<string, ActionKind>)[command] ?? null;
}
