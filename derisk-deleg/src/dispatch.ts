// Correspondance commande routeur -> type d'action. Fonction pure (testable
// sans mock d'api OpenClaw) ; le câblage réel (appel des fonctions delegate*)
// reste dans index.ts, qui est la seule couche à connaître `api`.

export type RouterCommand =
  | "/c" | "/q" | "/ingest" | "/wikistatus" | "/source"
  | "/chercher" | "/connecter" | "/inbox" | "/drive" | "/repondre";

export type ActionKind =
  | { kind: "wiki"; op: "capture" | "query" | "ingest" | "status" }
  | { kind: "scout" }
  | { kind: "gog"; op: "search" | "auth_start" | "inbox" | "drive_search" }
  | { kind: "gog-reply" };

const TABLE: Record<RouterCommand, ActionKind> = {
  "/c": { kind: "wiki", op: "capture" },
  "/q": { kind: "wiki", op: "query" },
  "/ingest": { kind: "wiki", op: "ingest" },
  "/wikistatus": { kind: "wiki", op: "status" },
  "/source": { kind: "scout" },
  "/chercher": { kind: "gog", op: "search" },
  "/connecter": { kind: "gog", op: "auth_start" },
  "/inbox": { kind: "gog", op: "inbox" },
  "/drive": { kind: "gog", op: "drive_search" },
  "/repondre": { kind: "gog-reply" },
};

export function commandToAction(command: string): ActionKind | null {
  return (TABLE as Record<string, ActionKind>)[command] ?? null;
}
