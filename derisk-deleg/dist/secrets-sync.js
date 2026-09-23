// Synchronisation des clés API depuis ~/.config/secrets.env vers les copies
// qui en dépendent (gateway.systemd.env, Wiki_LM/.env). Fonctions pures et
// testables sans toucher au disque ni au réseau ; le câblage (lecture des
// fichiers, écriture, redémarrage des services) reste dans index.ts, seule
// couche à connaître les chemins hôte et `api`.
//
// Garde-fou : incident du 22/09/2026 — un TELEGRAM_BOT_TOKEN de prod copié
// par erreur dans secrets.env sur sanroque (dev) a fait taire le bot dev
// pendant plusieurs heures sans erreur visible. validateTelegramToken
// vérifie via l'API Telegram elle-même que le token appartient bien au bot
// attendu pour la machine courante avant toute écriture.
export function parseEnv(content) {
    const out = {};
    for (const line of content.split("\n")) {
        const m = line.match(/^([A-Za-z_][A-Za-z0-9_]*)=(.*)$/);
        if (!m)
            continue;
        let value = m[2];
        if ((value.startsWith('"') && value.endsWith('"') && value.length >= 2) ||
            (value.startsWith("'") && value.endsWith("'") && value.length >= 2)) {
            value = value.slice(1, -1);
        }
        out[m[1]] = value;
    }
    return out;
}
// Ne retient que les clés présentes dans les deux fichiers et dont la valeur
// diffère — jamais d'ajout de clé absente du fichier cible (Wiki_LM/.env n'a
// pas à recevoir TELEGRAM_BOT_TOKEN, par exemple).
export function diffEnv(source, target) {
    const diffs = [];
    for (const key of Object.keys(target)) {
        if (key in source && source[key] !== target[key]) {
            diffs.push({ key, oldValue: target[key], newValue: source[key] });
        }
    }
    return diffs;
}
export function applyEnvDiff(content, diffs) {
    let out = content;
    for (const { key, newValue } of diffs) {
        const re = new RegExp(`^${key}=.*$`, "m");
        if (re.test(out)) {
            out = out.replace(re, `${key}=${newValue}`);
        }
    }
    return out;
}
// sanroque = bot dev, santiago = bot prod (cf. openclaw-config/INSTALL.md,
// section "Deux instances").
export const EXPECTED_TELEGRAM_BOT = {
    sanroque: "secretarius_tiron_bot",
    santiago: "secretarius1789_bot",
};
export async function validateTelegramToken(token, expectedUsername, fetchImpl = fetch) {
    let resp;
    try {
        resp = await fetchImpl(`https://api.telegram.org/bot${token}/getMe`, {
            signal: AbortSignal.timeout(10000),
        });
    }
    catch (e) {
        return { ok: false, reason: `getMe injoignable (${e?.message ?? e})` };
    }
    if (!resp.ok) {
        return { ok: false, reason: `getMe a échoué (HTTP ${resp.status})` };
    }
    const data = await resp.json();
    if (!data?.ok) {
        return { ok: false, reason: data?.description || "réponse Telegram invalide" };
    }
    const actualUsername = data.result?.username;
    if (actualUsername !== expectedUsername) {
        return {
            ok: false,
            reason: `token valide mais pour @${actualUsername}, attendu @${expectedUsername}`,
            actualUsername,
        };
    }
    return { ok: true, actualUsername };
}
