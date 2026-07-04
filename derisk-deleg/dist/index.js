import { Type } from "typebox";
import { definePluginEntry } from "openclaw/plugin-sdk/core";
import { parseReply } from "./parse.js";
import { commandToAction } from "./dispatch.js";
import { readFileSync, writeFileSync, existsSync, rmSync, readdirSync, statSync, copyFileSync, mkdirSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
const ROUTER_URL = "http://127.0.0.1:8999/route";
async function callRouter(message) {
    try {
        const resp = await fetch(ROUTER_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message }),
            signal: AbortSignal.timeout(30000),
        });
        if (!resp.ok)
            return { status: "unavailable" };
        const data = await resp.json();
        return data.status === "ok"
            ? { status: "ok", command: data.command, args: data.args ?? "" }
            : { status: "no_match" };
    }
    catch {
        return { status: "unavailable" };
    }
}
// Outils d'orchestration déterministes : une commande -> un outil -> délégation
// à l'agent wiki via api.runtime.subagent, avec le message "op: <op> | <arg>"
// construit par le code (routage ET contrat d'op déterministes).
//
// Règles (cf. spec) : ces outils doivent être `deny` pour les sous-agents
// (sinon le sous-agent les appelle au lieu d'exécuter son travail -> boucle).
// Générique : lance un sous-agent (sessionKey encode l'agent cible), attend,
// et relaie le dernier message assistant.
async function runAndRead(api, sessionKey, message) {
    const { runId } = await api.runtime.subagent.run({
        sessionKey,
        message,
        deliver: false,
    });
    const w = await api.runtime.subagent.waitForRun({ runId, timeoutMs: 120000 });
    const { messages } = await api.runtime.subagent.getSessionMessages({
        sessionKey,
        limit: 20,
    });
    const msgs = Array.isArray(messages) ? messages : [];
    const textOf = (m) => {
        const c = m?.content ?? m?.message?.content;
        if (typeof c === "string")
            return c;
        if (Array.isArray(c)) {
            return c
                .filter((b) => b?.type === "text" && b?.text)
                .map((b) => b.text)
                .join(" ");
        }
        return "";
    };
    let answer = "";
    for (const m of msgs) {
        const role = m?.role ?? m?.message?.role;
        const t = textOf(m);
        if (role === "assistant" && t.trim())
            answer = t;
    }
    const body = answer.trim() || JSON.stringify(msgs).slice(0, 900);
    return w?.status === "ok" ? body : `(run ${w?.status}) ${body}`;
}
// Suffixe unique : une session FRAÎCHE par délégation (sinon le sous-agent garde
// l'historique des appels précédents -> faux « déjà fait », contexte périmé).
function uniq() {
    return `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}
// Wiki : message "op: <op> | <arg>" vers l'agent wiki.
function delegateWiki(api, op, arg) {
    return runAndRead(api, `agent:wiki:subagent:cmd-${op}-${uniq()}`, `op: ${op} | ${arg}`.trimEnd());
}
// Scout : message "url: <url>" vers l'agent scout (lecture externe anti-injection).
function delegateScout(api, url) {
    return runAndRead(api, `agent:scout:subagent:cmd-source-${uniq()}`, `url: ${url}`);
}
// gog : message "op: <op> | <arg>" vers l'agent gog (Google via gog en sandbox).
function delegateGog(api, op, arg) {
    return runAndRead(api, `agent:gog:subagent:cmd-${op}-${uniq()}`, `op: ${op} | ${arg}`.trimEnd());
}
// Brouillon d'email en attente de confirmation (assistant mono-utilisateur).
// Le flux : gog_send prépare le brouillon (n'envoie pas) -> /confirm l'envoie via
// l'agent gog. Confirmation déterministe et Telegram-native, sans approbation native.
// Expire après PENDING_TTL_MS pour éviter qu'un /confirm tardif envoie un périmé.
const PENDING_TTL_MS = 10 * 60 * 1000;
let pending = null;
const AUTH_TTL_MS = 10 * 60 * 1000;
const GOG_CFG = `${process.env.HOME}/.openclaw/workspace/.gog-config`;
let pendingAuth = null;
export default definePluginEntry({
    id: "derisk-deleg",
    name: "Secretarius Wiki Commands",
    description: "Deterministic command tools delegating wiki ops (capture/status) to the wiki subagent.",
    register(api) {
        api.registerTool({
            name: "wiki_capture",
            description: "Capture a URL/note into the wiki inbox (delegates 'op: capture' to the wiki agent).",
            parameters: Type.Object({
                command: Type.Optional(Type.String({ description: "Raw args: #tags + URL or text to capture." })),
            }),
            async execute(_id, params) {
                let arg = (params?.command ?? "").trim();
                // Joindre le fichier le plus récent de media/inbound/ (< 5 min) si présent.
                const inboundDir = join(homedir(), ".openclaw", "media", "inbound");
                const attachDir = join(homedir(), "Documents", "Arbath", "Wiki_LM", "attachments");
                if (existsSync(inboundDir)) {
                    const now = Date.now();
                    const recent = readdirSync(inboundDir)
                        .map((f) => ({ f, mt: statSync(join(inboundDir, f)).mtimeMs }))
                        .filter(({ mt }) => now - mt < 5 * 60 * 1000)
                        .sort((a, b) => b.mt - a.mt)[0];
                    if (recent) {
                        mkdirSync(attachDir, { recursive: true });
                        const dest = join(attachDir, recent.f);
                        copyFileSync(join(inboundDir, recent.f), dest);
                        rmSync(join(inboundDir, recent.f));
                        arg += ` ref:${dest}`;
                    }
                }
                const out = await delegateWiki(api, "capture", arg);
                return { content: [{ type: "text", text: out.slice(0, 1800) }] };
            },
        });
        api.registerTool({
            name: "wiki_status",
            description: "Report wiki ingestion status (delegates 'op: status' to the wiki agent).",
            parameters: Type.Object({
                command: Type.Optional(Type.String({ description: "Unused." })),
            }),
            async execute(_id, _params) {
                const out = await delegateWiki(api, "status", "");
                return { content: [{ type: "text", text: out.slice(0, 1800) }] };
            },
        });
        api.registerTool({
            name: "wiki_ingest",
            description: "Process the wiki capture queue (delegates 'op: ingest' to the wiki agent; runs async).",
            parameters: Type.Object({
                command: Type.Optional(Type.String({ description: "Unused." })),
            }),
            async execute(_id, _params) {
                const out = await delegateWiki(api, "ingest", "");
                return { content: [{ type: "text", text: out.slice(0, 1800) }] };
            },
        });
        api.registerTool({
            name: "wiki_query",
            description: "Ask the wiki knowledge base a question (delegates 'op: query' to the wiki agent).",
            parameters: Type.Object({
                command: Type.Optional(Type.String({ description: "Raw args: the natural-language question." })),
            }),
            async execute(_id, params) {
                const arg = (params?.command ?? "").trim();
                const out = await delegateWiki(api, "query", arg);
                return { content: [{ type: "text", text: out.slice(0, 1800) }] };
            },
        });
        api.registerTool({
            name: "wiki_tags",
            description: "Liste les tags du wiki (délègue 'op: tags' à l'agent wiki).",
            parameters: Type.Object({ command: Type.Optional(Type.String({ description: "Inutilisé." })) }),
            async execute(_id, _params) {
                const out = await delegateWiki(api, "tags", "");
                return { content: [{ type: "text", text: out.slice(0, 1800) }] };
            },
        });
        api.registerTool({
            name: "wiki_kb_update",
            description: "Met à jour le KB depuis le dernier clustering (délègue 'op: kb_update' à l'agent wiki ; async).",
            parameters: Type.Object({ command: Type.Optional(Type.String({ description: "Inutilisé." })) }),
            async execute(_id, _params) {
                const out = await delegateWiki(api, "kb_update", "");
                return { content: [{ type: "text", text: out.slice(0, 1800) }] };
            },
        });
        api.registerTool({
            name: "source_read",
            description: "Read/summarize an external web page NOW via the anti-injection scout agent (delegates 'url: <url>').",
            parameters: Type.Object({
                command: Type.Optional(Type.String({ description: "Raw args: the URL to read." })),
            }),
            async execute(_id, params) {
                const url = (params?.command ?? "").trim();
                if (!url) {
                    return { content: [{ type: "text", text: "Usage: /source <url>" }] };
                }
                const out = await delegateScout(api, url);
                return { content: [{ type: "text", text: out.slice(0, 1800) }] };
            },
        });
        api.registerTool({
            name: "gog_inbox",
            description: "List recent inbox emails (delegates 'op: inbox' to the gog agent, which runs gog in its sandbox).",
            parameters: Type.Object({
                command: Type.Optional(Type.String({ description: "Optional Gmail query (default: in:inbox)." })),
            }),
            async execute(_id, params) {
                const q = (params?.command ?? "").trim();
                const out = q
                    ? await delegateGog(api, "search", q)
                    : await delegateGog(api, "inbox", "");
                return { content: [{ type: "text", text: out.slice(0, 1800) }] };
            },
        });
        api.registerTool({
            name: "gog_search",
            description: "Rechercher des emails Gmail (délègue 'op: search' à l'agent gog).",
            parameters: Type.Object({
                command: Type.Optional(Type.String({ description: "Requête Gmail." })),
            }),
            async execute(_id, params) {
                const q = (params?.command ?? "").trim();
                if (!q)
                    return { content: [{ type: "text", text: "Usage: /chercher <requête>" }] };
                const out = await delegateGog(api, "search", q);
                return { content: [{ type: "text", text: out.slice(0, 1800) }] };
            },
        });
        api.registerTool({
            name: "gog_get",
            description: "Lire le contenu d'un email Gmail par son id (délègue 'op: get' à l'agent gog).",
            parameters: Type.Object({
                command: Type.Optional(Type.String({ description: "L'id du message." })),
            }),
            async execute(_id, params) {
                const id = (params?.command ?? "").trim();
                if (!id)
                    return { content: [{ type: "text", text: "Usage: /lire <id>" }] };
                const out = await delegateGog(api, "get", id);
                return { content: [{ type: "text", text: out.slice(0, 1800) }] };
            },
        });
        api.registerTool({
            name: "gog_drive_search",
            description: "Rechercher des fichiers Google Drive (délègue 'op: drive_search' à l'agent gog).",
            parameters: Type.Object({
                command: Type.Optional(Type.String({ description: "Requête Drive." })),
            }),
            async execute(_id, params) {
                const q = (params?.command ?? "").trim();
                if (!q)
                    return { content: [{ type: "text", text: "Usage: /drive <requête>" }] };
                const out = await delegateGog(api, "drive_search", q);
                return { content: [{ type: "text", text: out.slice(0, 1800) }] };
            },
        });
        // gog_send : PRÉPARE un brouillon d'email (n'envoie PAS). main compose puis
        // appelle cet outil ; l'envoi réel se fait ensuite via /confirm.
        api.registerTool({
            name: "gog_send",
            description: "Prépare un brouillon d'email pour confirmation (n'envoie PAS). Après l'appel, indiquer à l'utilisateur de taper /confirm pour envoyer.",
            parameters: Type.Object({
                to: Type.String({ description: "Destinataire (email)." }),
                subject: Type.String({ description: "Sujet." }),
                body: Type.String({ description: "Corps en texte brut." }),
            }),
            async execute(_id, params) {
                pending = {
                    kind: "send",
                    to: params.to,
                    subject: params.subject,
                    body: params.body,
                    ts: Date.now(),
                };
                return {
                    content: [
                        {
                            type: "text",
                            text: `📧 Brouillon prêt (non envoyé) :\n• À : ${params.to}\n• Sujet : ${params.subject}\n• Corps : ${params.body}\n\nTapez /confirm pour envoyer (valable 10 min), ou /annuler pour abandonner.`,
                        },
                    ],
                };
            },
        });
        api.registerTool({
            name: "gog_reply",
            description: "Prépare un brouillon de RÉPONSE à un email (n'envoie PAS). Indiquer ensuite à l'utilisateur de taper /confirm.",
            parameters: Type.Object({
                command: Type.Optional(Type.String({ description: "Args bruts : <id> <texte de la réponse>." })),
            }),
            async execute(_id, params) {
                const parsed = parseReply(params?.command ?? "");
                if (!parsed) {
                    return { content: [{ type: "text", text: "Usage: /repondre <id> <texte>" }] };
                }
                pending = { kind: "reply", messageId: parsed.messageId, body: parsed.body, ts: Date.now() };
                return {
                    content: [
                        {
                            type: "text",
                            text: `📧 Brouillon de réponse prêt (non envoyé) :\n• En réponse à : ${parsed.messageId}\n• Corps : ${parsed.body}\n\nTapez /confirm pour envoyer (valable 10 min), ou /annuler pour abandonner.`,
                        },
                    ],
                };
            },
        });
        api.registerTool({
            name: "gog_connect_start",
            description: "Démarre l'autorisation Google (délègue 'op: auth_start' à l'agent gog), puis renvoie l'URL d'autorisation à coller dans le navigateur.",
            parameters: Type.Object({
                command: Type.Optional(Type.String({ description: "Inutilisé." })),
            }),
            async execute(_id, _params) {
                try {
                    rmSync(`${GOG_CFG}/auth_url`);
                }
                catch { }
                try {
                    rmSync(`${GOG_CFG}/auth_response`);
                }
                catch { }
                try {
                    rmSync(`${GOG_CFG}/auth_done`);
                }
                catch { }
                await delegateGog(api, "auth_start", "");
                // Attendre que le bridge expose l'URL (poll court).
                let url = "";
                for (let i = 0; i < 60; i++) {
                    if (existsSync(`${GOG_CFG}/auth_url`)) {
                        url = readFileSync(`${GOG_CFG}/auth_url`, "utf8").trim();
                        if (url)
                            break;
                    }
                    await new Promise((r) => setTimeout(r, 500));
                }
                if (!url) {
                    return { content: [{ type: "text", text: "Échec : URL d'autorisation non générée. Réessayez /connecter." }] };
                }
                pendingAuth = { ts: Date.now() };
                return {
                    content: [
                        {
                            type: "text",
                            text: `Pour connecter votre compte Google :\n1. Ouvrez ce lien et autorisez :\n${url}\n2. Recollez ici l'URL de redirection obtenue après autorisation.`,
                        },
                    ],
                };
            },
        });
        // /confirm et /annuler : gérés au niveau MESSAGE (hook inbound_claim), AVANT
        // le modèle. Le modèle n'a donc aucun outil confirm/cancel à appeler (fini les
        // appels erratiques). L'utilisateur seul déclenche ces commandes ; le hook
        // exécute l'action, répond, et réclame le message (handled:true => pas de modèle).
        api.on("before_agent_reply", async (event) => {
            const text = String(event?.cleanedBody ?? "");
            // Retour OAuth : si une autorisation est en attente, le message courant est
            // l'URL de redirection à injecter dans le pont gog.
            if (pendingAuth && !/^\s*\//.test(text) && text.trim()) {
                if (Date.now() - pendingAuth.ts > AUTH_TTL_MS) {
                    pendingAuth = null;
                    return { handled: true, reply: { text: "Autorisation expirée — relancez /connecter." } };
                }
                pendingAuth = null;
                writeFileSync(`${GOG_CFG}/auth_response`, text.trim(), "utf8");
                let done = "";
                for (let i = 0; i < 60; i++) {
                    if (existsSync(`${GOG_CFG}/auth_done`)) {
                        done = readFileSync(`${GOG_CFG}/auth_done`, "utf8").trim();
                        break;
                    }
                    await new Promise((r) => setTimeout(r, 500));
                }
                const ok = done === "ok";
                return { handled: true, reply: { text: ok ? "Compte Google connecté." : `Échec de la connexion (${done || "timeout"}). Réessayez /connecter.` } };
            }
            const m = text.match(/(^|\s)\/(confirm|annuler)\b/i);
            const cmd = m ? "/" + m[2].toLowerCase() : "";
            if (cmd === "/confirm") {
                if (!pending) {
                    return { handled: true, reply: { text: "Rien à confirmer (aucun brouillon en attente)." } };
                }
                if (Date.now() - pending.ts > PENDING_TTL_MS) {
                    pending = null;
                    return { handled: true, reply: { text: "Brouillon expiré (plus de 10 min) — rien envoyé. Recomposez si besoin." } };
                }
                const p = pending;
                pending = null;
                const out = p.kind === "send"
                    ? await delegateGog(api, "send", `to=${p.to}; subject=${p.subject}; body=${p.body}`)
                    : await delegateGog(api, "reply", `id=${p.messageId}; body=${p.body}`);
                return { handled: true, reply: { text: out.slice(0, 1800) } };
            }
            if (cmd === "/annuler") {
                if (!pending) {
                    return { handled: true, reply: { text: "Rien à annuler (aucun brouillon en attente)." } };
                }
                const dest = pending.kind === "send" ? pending.to : `réponse à ${pending.messageId}`;
                pending = null;
                return {
                    handled: true,
                    reply: { text: `Brouillon abandonné (était destiné à ${dest}). Rien n'a été envoyé.` },
                };
            }
            // Routage déterministe via le service Tiron : tout message qui n'est
            // pas déjà /confirm, /annuler, ou un retour OAuth.
            const routed = await callRouter(text);
            if (routed.status === "unavailable") {
                return { handled: true, reply: { text: "Routeur local indisponible, réessayez dans un instant." } };
            }
            if (routed.status === "no_match") {
                return {
                    handled: true,
                    reply: { text: "Je n'ai pas identifié de commande (essayez /q <question>, /c <url>...)." },
                };
            }
            const action = commandToAction(routed.command);
            if (action === null) {
                return {
                    handled: true,
                    reply: { text: "Je n'ai pas identifié de commande (essayez /q <question>, /c <url>...)." },
                };
            }
            if (action.kind === "wiki") {
                const out = await delegateWiki(api, action.op, routed.args);
                return { handled: true, reply: { text: out.slice(0, 1800) } };
            }
            if (action.kind === "scout") {
                const out = await delegateScout(api, routed.args.trim());
                return { handled: true, reply: { text: out.slice(0, 1800) } };
            }
            if (action.kind === "gog") {
                const out = await delegateGog(api, action.op, routed.args);
                return { handled: true, reply: { text: out.slice(0, 1800) } };
            }
            // action.kind === "gog-reply" : réutilise EXACTEMENT la logique de mise
            // en attente existante (parseReply + pending), jamais de délégation
            // directe — /repondre est la seule commande sensible atteignable ici.
            const parsed = parseReply(routed.args);
            if (!parsed) {
                return { handled: true, reply: { text: "Usage: /repondre <id> <texte>" } };
            }
            pending = { kind: "reply", messageId: parsed.messageId, body: parsed.body, ts: Date.now() };
            return {
                handled: true,
                reply: {
                    text: `📧 Brouillon de réponse prêt (non envoyé) :\n• En réponse à : ${parsed.messageId}\n• Corps : ${parsed.body}\n\nTapez /confirm pour envoyer (valable 10 min), ou /annuler pour abandonner.`,
                },
            };
        });
        // Garde-fou déterministe : un `gog` d'écriture sensible (send/delete/share/…)
        // ne peut s'exécuter QUE dans l'agent gog (atteint uniquement via /confirm).
        // Tout envoi direct ailleurs (ex. par main) est BLOQUÉ.
        api.on("before_tool_call", async (event, ctx) => {
            if (event?.toolName !== "exec")
                return;
            const cmd = String(event?.params?.command ?? "");
            if (!/\bgog\b/.test(cmd))
                return;
            const isSensitive = /\b(send|reply|forward|delete|trash|remove|share|revoke|rm)\b/i.test(cmd);
            if (!isSensitive)
                return;
            const agentId = ctx?.agentId ?? event?.agentId ?? event?.ctx?.agentId;
            if (agentId === "gog")
                return; // flux confirmé (/confirm -> agent gog) : autorisé
            return {
                block: true,
                blockReason: "Envoi/écriture Google direct interdit. Composez puis appelez l'outil gog_send (prépare le brouillon), et l'utilisateur tapera /confirm pour envoyer.",
            };
        }, { priority: 50 });
    },
});
