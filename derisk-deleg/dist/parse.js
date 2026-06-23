// Helpers purs (testables) pour les commandes du plugin derisk-deleg.
// /repondre <id> <texte> : premier token = id, reste = corps (espaces conservés).
export function parseReply(raw) {
    const s = (raw ?? "").trim();
    const sp = s.indexOf(" ");
    if (sp < 1)
        return null;
    const messageId = s.slice(0, sp);
    const body = s.slice(sp + 1).trim();
    if (!messageId || !body)
        return null;
    return { messageId, body };
}
