// Correspondance commande routeur -> type d'action. Fonction pure (testable
// sans mock d'api OpenClaw) ; le câblage réel (appel des fonctions delegate*)
// reste dans index.ts, qui est la seule couche à connaître `api`.
const TABLE = {
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
    "/r": { kind: "wiki", op: "search" },
    "/tags": { kind: "wiki", op: "tags" },
    "/kbupdate": { kind: "wiki", op: "kb_update" },
    "/lire": { kind: "gog", op: "get" },
    "/supprimer": { kind: "wiki", op: "delete" },
    "/relire": { kind: "wiki", op: "review" },
    "/verifie": { kind: "wiki", op: "verify" },
};
export function commandToAction(command) {
    return TABLE[command] ?? null;
}
