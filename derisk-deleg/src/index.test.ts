import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { existsSync, readFileSync, writeFileSync, rmSync } from "node:fs";

// Seules les 4 fonctions réellement utilisées par le flux GOG_CFG/OAuth sont
// mockées ; le reste de node:fs (readdirSync, statSync, copyFileSync,
// mkdirSync — utilisées par wiki_capture, non testé ici) reste réel.
vi.mock("node:fs", async (importOriginal) => {
  const actual = await importOriginal<typeof import("node:fs")>();
  return {
    ...actual,
    existsSync: vi.fn(),
    readFileSync: vi.fn(),
    writeFileSync: vi.fn(),
    rmSync: vi.fn(),
  };
});

// --- Harnais : capture ce que register(api) enregistre, sans toucher à
// index.ts. api.runtime.subagent.{run,waitForRun,getSessionMessages} sont les
// seuls points d'entrée réseau/sandbox du plugin pour gog/scout.
function makeApi(overrides: {
  run?: any;
  waitForRun?: any;
  getSessionMessages?: any;
} = {}) {
  const tools: Record<string, any> = {};
  const hooks: Record<string, { handler: Function; opts?: any }> = {};
  const run =
    overrides.run ?? vi.fn(async () => ({ runId: "r1" }));
  const waitForRun =
    overrides.waitForRun ?? vi.fn(async () => ({ status: "ok" }));
  const getSessionMessages =
    overrides.getSessionMessages ??
    vi.fn(async () => ({
      messages: [{ role: "assistant", content: "réponse simulée" }],
    }));
  const api = {
    registerTool(def: any) {
      tools[def.name] = def;
    },
    on(event: string, handler: Function, opts?: any) {
      hooks[event] = { handler, opts };
    },
    runtime: { subagent: { run, waitForRun, getSessionMessages } },
  };
  return { api, tools, hooks, run, waitForRun, getSessionMessages };
}

// pending/pendingAuth sont des variables de module dans index.ts : réimporter
// à froid à chaque test isole complètement leur état (au prix d'un import par
// test — mesuré < 100 ms à chaud dans cette suite).
async function freshPlugin() {
  vi.resetModules();
  const mod: any = await import("./index.js");
  return mod.default as { register: (api: any) => void };
}

beforeEach(() => {
  // Les mocks node:fs sont créés une fois par le factory de vi.mock (pas
  // réinitialisés par vi.resetModules()) : les nettoyer explicitement pour
  // isoler chaque test (sinon l'historique d'appels d'un test OAuth déborde
  // sur le suivant).
  vi.mocked(existsSync).mockReset();
  vi.mocked(readFileSync).mockReset();
  vi.mocked(writeFileSync).mockReset();
  vi.mocked(rmSync).mockReset();
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

describe("outils gog_* — délégation au sous-agent gog", () => {
  it("gog_inbox sans argument délègue op:inbox", async () => {
    const plugin = await freshPlugin();
    const { api, tools, run } = makeApi();
    plugin.register(api);

    const res = await tools["gog_inbox"].execute("id", {});

    expect(res.content[0].text).toBe("réponse simulée");
    expect(run).toHaveBeenCalledTimes(1);
    const [{ sessionKey, message, deliver }] = run.mock.calls[0];
    expect(sessionKey).toMatch(/^agent:gog:subagent:cmd-inbox-/);
    expect(message).toBe("op: inbox |");
    expect(deliver).toBe(false);
  });

  it("gog_inbox avec argument délègue op:search (pas op:inbox)", async () => {
    const plugin = await freshPlugin();
    const { api, tools, run } = makeApi();
    plugin.register(api);

    await tools["gog_inbox"].execute("id", { command: "de:x@y.com" });

    const [{ sessionKey, message }] = run.mock.calls[0];
    expect(sessionKey).toMatch(/^agent:gog:subagent:cmd-search-/);
    expect(message).toBe("op: search | de:x@y.com");
  });

  it("gog_search sans argument renvoie l'usage sans déléguer", async () => {
    const plugin = await freshPlugin();
    const { api, tools, run } = makeApi();
    plugin.register(api);

    const res = await tools["gog_search"].execute("id", {});

    expect(res.content[0].text).toBe("Usage: /chercher <requête>");
    expect(run).not.toHaveBeenCalled();
  });

  it("gog_get délègue op:get avec l'id du message", async () => {
    const plugin = await freshPlugin();
    const { api, tools, run } = makeApi();
    plugin.register(api);

    await tools["gog_get"].execute("id", { command: "18ab" });

    const [{ message }] = run.mock.calls[0];
    expect(message).toBe("op: get | 18ab");
  });

  it("gog_drive_search délègue op:drive_search", async () => {
    const plugin = await freshPlugin();
    const { api, tools, run } = makeApi();
    plugin.register(api);

    await tools["gog_drive_search"].execute("id", { command: "rapport 2026" });

    const [{ message }] = run.mock.calls[0];
    expect(message).toBe("op: drive_search | rapport 2026");
  });

  it("un run non 'ok' préfixe la réponse par (run <status>)", async () => {
    const plugin = await freshPlugin();
    const { api, tools, run } = makeApi({
      waitForRun: vi.fn(async () => ({ status: "timeout" })),
    });
    plugin.register(api);

    const res = await tools["gog_get"].execute("id", { command: "18ab" });

    expect(res.content[0].text).toBe("(run timeout) réponse simulée");
    expect(run).toHaveBeenCalledTimes(1);
  });

  it("gog_send prépare un brouillon SANS déléguer", async () => {
    const plugin = await freshPlugin();
    const { api, tools, run } = makeApi();
    plugin.register(api);

    const res = await tools["gog_send"].execute("id", {
      to: "a@b.com",
      subject: "Sujet",
      body: "Corps",
    });

    expect(run).not.toHaveBeenCalled();
    const text = res.content[0].text as string;
    expect(text).toContain("📧 Brouillon prêt (non envoyé) :");
    expect(text).toContain("À : a@b.com");
    expect(text).toContain("Sujet : Sujet");
    expect(text).toContain("Corps : Corps");
    expect(text).toContain("/confirm");
    expect(text).toContain("/annuler");
  });

  it("gog_reply avec une commande invalide (pas de corps) renvoie l'usage", async () => {
    const plugin = await freshPlugin();
    const { api, tools } = makeApi();
    plugin.register(api);

    const res = await tools["gog_reply"].execute("id", { command: "18ab" });

    expect(res.content[0].text).toBe("Usage: /repondre <id> <texte>");
  });

  it("gog_reply valide prépare un brouillon de réponse SANS déléguer", async () => {
    const plugin = await freshPlugin();
    const { api, tools, run } = makeApi();
    plugin.register(api);

    const res = await tools["gog_reply"].execute("id", {
      command: "18ab voici la réponse",
    });

    expect(run).not.toHaveBeenCalled();
    const text = res.content[0].text as string;
    expect(text).toContain("En réponse à : 18ab");
    expect(text).toContain("Corps : voici la réponse");
  });
});

describe("source_read — délégation à scout", () => {
  it("délègue url:<url> à l'agent scout", async () => {
    const plugin = await freshPlugin();
    const { api, tools, run } = makeApi();
    plugin.register(api);

    const res = await tools["source_read"].execute("id", {
      command: "https://example.com/a",
    });

    expect(res.content[0].text).toBe("réponse simulée");
    const [{ sessionKey, message }] = run.mock.calls[0];
    expect(sessionKey).toMatch(/^agent:scout:subagent:cmd-source-/);
    expect(message).toBe("url: https://example.com/a");
  });

  it("sans url renvoie l'usage sans déléguer", async () => {
    const plugin = await freshPlugin();
    const { api, tools, run } = makeApi();
    plugin.register(api);

    const res = await tools["source_read"].execute("id", {});

    expect(res.content[0].text).toBe("Usage: /source <url>");
    expect(run).not.toHaveBeenCalled();
  });
});

describe("before_agent_reply — /confirm et /annuler", () => {
  it("/confirm sans brouillon en attente", async () => {
    const plugin = await freshPlugin();
    const { api, hooks } = makeApi();
    plugin.register(api);

    const res = await hooks["before_agent_reply"].handler({ cleanedBody: "/confirm" });

    expect(res).toEqual({
      handled: true,
      reply: { text: "Rien à confirmer (aucun brouillon en attente)." },
    });
  });

  it("/confirm avec un brouillon 'send' en attente délègue op:send et le vide", async () => {
    const plugin = await freshPlugin();
    const { api, tools, hooks, run } = makeApi();
    plugin.register(api);

    await tools["gog_send"].execute("id", { to: "a@b.com", subject: "Sujet", body: "Corps" });
    const res = await hooks["before_agent_reply"].handler({ cleanedBody: "/confirm" });

    expect(res).toEqual({ handled: true, reply: { text: "réponse simulée" } });
    const [{ sessionKey, message }] = run.mock.calls[0];
    expect(sessionKey).toMatch(/^agent:gog:subagent:cmd-send-/);
    expect(message).toBe("op: send | to=a@b.com; subject=Sujet; body=Corps");

    // le brouillon est consommé : un second /confirm ne trouve plus rien
    const res2 = await hooks["before_agent_reply"].handler({ cleanedBody: "/confirm" });
    expect(res2.reply.text).toBe("Rien à confirmer (aucun brouillon en attente).");
  });

  it("/confirm après expiration (>10 min) refuse et vide le brouillon", async () => {
    vi.useFakeTimers();
    const plugin = await freshPlugin();
    const { api, tools, hooks, run } = makeApi();
    plugin.register(api);

    await tools["gog_send"].execute("id", { to: "a@b.com", subject: "s", body: "b" });
    vi.advanceTimersByTime(10 * 60 * 1000 + 1);
    const res = await hooks["before_agent_reply"].handler({ cleanedBody: "/confirm" });

    expect(res).toEqual({
      handled: true,
      reply: { text: "Brouillon expiré (plus de 10 min) — rien envoyé. Recomposez si besoin." },
    });
    expect(run).not.toHaveBeenCalled();
  });

  it("/annuler sans brouillon en attente", async () => {
    const plugin = await freshPlugin();
    const { api, hooks } = makeApi();
    plugin.register(api);

    const res = await hooks["before_agent_reply"].handler({ cleanedBody: "/annuler" });

    expect(res).toEqual({
      handled: true,
      reply: { text: "Rien à annuler (aucun brouillon en attente)." },
    });
  });

  it("/annuler avec un brouillon 'reply' en attente ne délègue jamais", async () => {
    const plugin = await freshPlugin();
    const { api, tools, hooks, run } = makeApi();
    plugin.register(api);

    await tools["gog_reply"].execute("id", { command: "18ab corps" });
    const res = await hooks["before_agent_reply"].handler({ cleanedBody: "/annuler" });

    expect(res).toEqual({
      handled: true,
      reply: { text: "Brouillon abandonné (était destiné à réponse à 18ab). Rien n'a été envoyé." },
    });
    expect(run).not.toHaveBeenCalled();

    const res2 = await hooks["before_agent_reply"].handler({ cleanedBody: "/confirm" });
    expect(res2.reply.text).toBe("Rien à confirmer (aucun brouillon en attente).");
  });
});

describe("before_agent_reply — retour OAuth Google", () => {
  it("connexion réussie (auth_done = ok)", async () => {
    const plugin = await freshPlugin();
    const { api, tools, hooks } = makeApi();
    plugin.register(api);

    vi.mocked(existsSync).mockImplementation((p: any) => String(p).endsWith("auth_url"));
    vi.mocked(readFileSync).mockReturnValue("https://accounts.google.com/o/oauth2/auth");
    await tools["gog_connect_start"].execute("id", {});

    vi.mocked(existsSync).mockImplementation((p: any) => String(p).endsWith("auth_done"));
    vi.mocked(readFileSync).mockReturnValue("ok");

    const res = await hooks["before_agent_reply"].handler({
      cleanedBody: "http://localhost:1/callback?code=abc",
    });

    expect(res).toEqual({ handled: true, reply: { text: "Compte Google connecté." } });
    expect(vi.mocked(writeFileSync)).toHaveBeenCalledWith(
      expect.stringContaining("auth_response"),
      "http://localhost:1/callback?code=abc",
      "utf8",
    );
  });

  it("connexion refusée (auth_done ≠ ok)", async () => {
    const plugin = await freshPlugin();
    const { api, tools, hooks } = makeApi();
    plugin.register(api);

    vi.mocked(existsSync).mockImplementation((p: any) => String(p).endsWith("auth_url"));
    vi.mocked(readFileSync).mockReturnValue("https://accounts.google.com/o/oauth2/auth");
    await tools["gog_connect_start"].execute("id", {});

    vi.mocked(existsSync).mockImplementation((p: any) => String(p).endsWith("auth_done"));
    vi.mocked(readFileSync).mockReturnValue("denied");

    const res = await hooks["before_agent_reply"].handler({ cleanedBody: "http://localhost:1/callback?err" });

    expect(res).toEqual({
      handled: true,
      reply: { text: "Échec de la connexion (denied). Réessayez /connecter." },
    });
  });

  it("retour OAuth après expiration (>10 min) : refusé sans écrire de fichier", async () => {
    vi.useFakeTimers();
    const plugin = await freshPlugin();
    const { api, tools, hooks } = makeApi();
    plugin.register(api);

    vi.mocked(existsSync).mockImplementation((p: any) => String(p).endsWith("auth_url"));
    vi.mocked(readFileSync).mockReturnValue("https://accounts.google.com/o/oauth2/auth");
    await tools["gog_connect_start"].execute("id", {});

    vi.advanceTimersByTime(10 * 60 * 1000 + 1);
    const res = await hooks["before_agent_reply"].handler({ cleanedBody: "http://localhost:1/callback" });

    expect(res).toEqual({
      handled: true,
      reply: { text: "Autorisation expirée — relancez /connecter." },
    });
    expect(vi.mocked(writeFileSync)).not.toHaveBeenCalled();
  });
});

describe("before_agent_reply — routage via tiron-router", () => {
  it("routeur indisponible (fetch en échec)", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => { throw new Error("ECONNREFUSED"); }));
    const plugin = await freshPlugin();
    const { api, hooks } = makeApi();
    plugin.register(api);

    const res = await hooks["before_agent_reply"].handler({ cleanedBody: "bonjour" });

    expect(res).toEqual({
      handled: true,
      reply: { text: "Routeur local indisponible, réessayez dans un instant." },
    });
  });

  it("réponse FAQ directe (status: answer)", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({ ok: true, json: async () => ({ status: "answer", reply: "Réponse FAQ." }) })),
    );
    const plugin = await freshPlugin();
    const { api, hooks } = makeApi();
    plugin.register(api);

    const res = await hooks["before_agent_reply"].handler({ cleanedBody: "quelle est l'IP de sanroque ?" });

    expect(res).toEqual({ handled: true, reply: { text: "Réponse FAQ." } });
  });

  it("no_match : message déterministe, aucune délégation", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({ ok: true, json: async () => ({ status: "no_match" }) })),
    );
    const plugin = await freshPlugin();
    const { api, hooks, run } = makeApi();
    plugin.register(api);

    const res = await hooks["before_agent_reply"].handler({ cleanedBody: "raconte une blague" });

    expect(res).toEqual({
      handled: true,
      reply: { text: "Je n'ai pas cette information (essayez /q <question>, /c <url>...)." },
    });
    expect(run).not.toHaveBeenCalled();
  });

  it("commande reconnue par le routeur mais absente de la table -> message déterministe", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({ ok: true, json: async () => ({ status: "ok", command: "/inexistant", args: "" }) })),
    );
    const plugin = await freshPlugin();
    const { api, hooks } = makeApi();
    plugin.register(api);

    const res = await hooks["before_agent_reply"].handler({ cleanedBody: "texte quelconque" });

    expect(res).toEqual({
      handled: true,
      reply: { text: "Je n'ai pas identifié de commande (essayez /q <question>, /c <url>...)." },
    });
  });

  it("commande routée vers scout (/source) délègue url:<args>", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({
        ok: true,
        json: async () => ({ status: "ok", command: "/source", args: " https://exemple.com " }),
      })),
    );
    const plugin = await freshPlugin();
    const { api, hooks, run } = makeApi();
    plugin.register(api);

    const res = await hooks["before_agent_reply"].handler({ cleanedBody: "regarde cette page https://exemple.com" });

    expect(res).toEqual({ handled: true, reply: { text: "réponse simulée" } });
    const [{ sessionKey, message }] = run.mock.calls[0];
    expect(sessionKey).toMatch(/^agent:scout:subagent:cmd-source-/);
    expect(message).toBe("url: https://exemple.com"); // args.trim()
  });

  it("commande routée vers gog (/chercher) délègue op:search", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({
        ok: true,
        json: async () => ({ status: "ok", command: "/chercher", args: "facture edf" }),
      })),
    );
    const plugin = await freshPlugin();
    const { api, hooks, run } = makeApi();
    plugin.register(api);

    const res = await hooks["before_agent_reply"].handler({ cleanedBody: "cherche la facture edf" });

    expect(res).toEqual({ handled: true, reply: { text: "réponse simulée" } });
    const [{ message }] = run.mock.calls[0];
    expect(message).toBe("op: search | facture edf");
  });

  it("commande routée vers /repondre (gog-reply) prépare un brouillon SANS déléguer", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({
        ok: true,
        json: async () => ({ status: "ok", command: "/repondre", args: "18ab corps de la réponse" }),
      })),
    );
    const plugin = await freshPlugin();
    const { api, hooks, run } = makeApi();
    plugin.register(api);

    const res = await hooks["before_agent_reply"].handler({ cleanedBody: "réponds à 18ab : corps de la réponse" });

    expect(run).not.toHaveBeenCalled();
    const text = (res as any).reply.text as string;
    expect(text).toContain("En réponse à : 18ab");
    expect(text).toContain("Corps : corps de la réponse");

    // le brouillon posé par le routage est bien le même pending que /confirm consulte
    const confirmRes = await hooks["before_agent_reply"].handler({ cleanedBody: "/confirm" });
    expect(confirmRes.reply.text).toBe("réponse simulée");
    const [{ sessionKey, message }] = run.mock.calls[0];
    expect(sessionKey).toMatch(/^agent:gog:subagent:cmd-reply-/);
    expect(message).toBe("op: reply | id=18ab; body=corps de la réponse");
  });

  it("commande routée vers /repondre avec args invalides (pas de corps) renvoie l'usage", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({
        ok: true,
        json: async () => ({ status: "ok", command: "/repondre", args: "18ab" }),
      })),
    );
    const plugin = await freshPlugin();
    const { api, hooks } = makeApi();
    plugin.register(api);

    const res = await hooks["before_agent_reply"].handler({ cleanedBody: "réponds à 18ab" });

    expect(res).toEqual({
      handled: true,
      reply: { text: "Usage: /repondre <id> <texte>" },
    });
  });
});

describe("before_tool_call — garde-fou gog (écriture directe bloquée)", () => {
  it("est enregistré avec la priorité 50", async () => {
    const plugin = await freshPlugin();
    const { api, hooks } = makeApi();
    plugin.register(api);

    expect(hooks["before_tool_call"].opts).toEqual({ priority: 50 });
  });

  it("bloque un exec 'gog ... send' hors agent gog", async () => {
    const plugin = await freshPlugin();
    const { api, hooks } = makeApi();
    plugin.register(api);

    const res = await hooks["before_tool_call"].handler(
      { toolName: "exec", params: { command: "gog email send --to a@b.com" } },
      { agentId: "main" },
    );

    expect(res).toEqual({
      block: true,
      blockReason:
        "Envoi/écriture Google direct interdit. Composez puis appelez l'outil gog_send (prépare le brouillon), et l'utilisateur tapera /confirm pour envoyer.",
    });
  });

  it("autorise le même exec depuis l'agent gog (flux /confirm)", async () => {
    const plugin = await freshPlugin();
    const { api, hooks } = makeApi();
    plugin.register(api);

    const res = await hooks["before_tool_call"].handler(
      { toolName: "exec", params: { command: "gog email send --to a@b.com" } },
      { agentId: "gog" },
    );

    expect(res).toBeUndefined();
  });

  it("laisse passer une lecture gog (aucun verbe sensible)", async () => {
    const plugin = await freshPlugin();
    const { api, hooks } = makeApi();
    plugin.register(api);

    const res = await hooks["before_tool_call"].handler(
      { toolName: "exec", params: { command: "gog email search foo" } },
      { agentId: "main" },
    );

    expect(res).toBeUndefined();
  });

  it("ignore un outil autre que exec", async () => {
    const plugin = await freshPlugin();
    const { api, hooks } = makeApi();
    plugin.register(api);

    const res = await hooks["before_tool_call"].handler(
      { toolName: "read", params: { command: "gog email send" } },
      { agentId: "main" },
    );

    expect(res).toBeUndefined();
  });

  it("ignore une commande exec sans 'gog'", async () => {
    const plugin = await freshPlugin();
    const { api, hooks } = makeApi();
    plugin.register(api);

    const res = await hooks["before_tool_call"].handler(
      { toolName: "exec", params: { command: "cat foo.txt" } },
      { agentId: "main" },
    );

    expect(res).toBeUndefined();
  });
});
