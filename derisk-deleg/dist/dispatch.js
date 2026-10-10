// Correspondance commande routeur -> type d'action. Fonction pure (testable
// sans mock d'api OpenClaw) ; le câblage réel (appel des fonctions delegate*)
// reste dans index.ts, qui est la seule couche à connaître `api`.
const TABLE = {
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
export function commandToAction(command) {
    return TABLE[command] ?? null;
}
