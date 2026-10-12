import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { existsSync, readFileSync, writeFileSync, rmSync } from "node:fs";
import { execSync } from "node:child_process";

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
    renameSync: vi.fn(),
    rmSync: vi.fn(),
  };
});

// /maj (secrets-sync) est la seule chose du plugin qui lance des commandes
// système (systemctl, systemd-run) : jamais réellement exécutées en test.
vi.mock("node:child_process", async (importOriginal) => {
  const actual = await importOriginal<typeof import("node:child_process")>();
  return { ...actual, execSync: vi.fn() };
});

// --- Harnais : capture ce que register(api) enregistre, sans toucher à
// index.ts. api.runtime.subagent.{run,waitForRun,getSessionMessages} sont les
// seuls points d'entrée réseau/sandbox du plugin pour gog/scout.
function makeApi(overrides: {
  run?: any;
  waitForRun?: any;
  getSessionMessages?: any;
  pluginConfig?: Record<string, unknown>;
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
    pluginConfig: overrides.pluginConfig,
    config: { agents: { entries: { gog: { sandbox: { docker: { env: { GOG_ACCOUNT: "test@gmail.com" } } } } } } },
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
  vi.mocked(execSync).mockReset();
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
  vi.useRealTimers();
  vi.doUnmock("./wiki-ops.js");
  vi.doUnmock("./cron-ops.js");
  vi.doUnmock("./gog-ops.js");
  vi.doUnmock("./mail-ops.js");
});

describe("outils gog_drive_search et source_read", () => {
  it("gog_drive_search appelle gog directement, sans sous-agent", async () => {
    const gogDriveSearch = vi.fn(async () => "1. rapport");
    vi.doMock("./gog-ops.js", () => ({ gogDriveSearch, gogAuthStart: vi.fn() }));
    const plugin = await freshPlugin();
    const { api, tools, run } = makeApi();
    plugin.register(api);

    const r = await tools["gog_drive_search"].execute("id", { command: "rapport 2026" });

    expect(gogDriveSearch).toHaveBeenCalledWith(api, "rapport 2026");
    expect(r.content[0].text).toBe("1. rapport");
    expect(run).not.toHaveBeenCalled();
  });

  it("un run non 'ok' préfixe la réponse par (run <status>)", async () => {
    const plugin = await freshPlugin();
    const { api, tools, run } = makeApi({
      waitForRun: vi.fn(async () => ({ status: "timeout" })),
    });
    plugin.register(api);

    const res = await tools["source_read"].execute("id", { command: "https://ex.fr" });

    expect(res.content[0].text).toBe("(run timeout) réponse simulée");
    expect(run).toHaveBeenCalledTimes(1);
  });

});

describe("outils wiki_review / wiki_verify / wiki_delete", () => {
  afterEach(() => {
    vi.doUnmock("./wiki-ops.js");
  });

  it("wiki_review délègue op:review sans argument", async () => {
    const runWikiOpSpy = vi.fn(async () => "Page à relire : src-a.");
    vi.doMock("./wiki-ops.js", () => ({ runWikiOp: runWikiOpSpy }));
    const plugin = await freshPlugin();
    const { api, tools } = makeApi();
    plugin.register(api);

    const res = await tools["wiki_review"].execute("id", {});

    expect(res.content[0].text).toBe("Page à relire : src-a.");
    expect(runWikiOpSpy).toHaveBeenCalledWith(api, "review", "");
  });

  it("wiki_verify sans argument renvoie l'usage sans rien préparer", async () => {
    const plugin = await freshPlugin();
    const { api, tools } = makeApi();
    plugin.register(api);

    const res = await tools["wiki_verify"].execute("id", {});

    expect(res.content[0].text).toBe("Usage: /verifie <page>");
  });

  it("wiki_verify prépare la confirmation SANS marquer directement", async () => {
    const runWikiOpSpy = vi.fn(async () => "ne doit jamais être appelé ici");
    vi.doMock("./wiki-ops.js", () => ({ runWikiOp: runWikiOpSpy }));
    const plugin = await freshPlugin();
    const { api, tools } = makeApi();
    plugin.register(api);

    const res = await tools["wiki_verify"].execute("id", { command: "src-a" });

    expect(runWikiOpSpy).not.toHaveBeenCalled();
    const text = res.content[0].text as string;
    expect(text).toContain("src-a");
    expect(text).toContain("/confirm");
    expect(text).toContain("/annuler");
  });

  it("wiki_delete sans argument renvoie l'usage sans essai à blanc", async () => {
    const fetchWikiOpJsonSpy = vi.fn(async () => ({ ok: true, text: "ne doit jamais être appelé ici" }));
    vi.doMock("./wiki-ops.js", () => ({ fetchWikiOpJson: fetchWikiOpJsonSpy }));
    const plugin = await freshPlugin();
    const { api, tools } = makeApi();
    plugin.register(api);

    const res = await tools["wiki_delete"].execute("id", {});

    expect(res.content[0].text).toBe("Usage: /supprimer <page>");
    expect(fetchWikiOpJsonSpy).not.toHaveBeenCalled();
  });

  it("wiki_delete : essai à blanc réussi prépare /confirm, ne supprime jamais directement", async () => {
    const fetchWikiOpJsonSpy = vi.fn(async () => ({
      ok: true,
      text: "Suppression de src-a : 2 pages affectées.",
    }));
    const runWikiOpSpy = vi.fn(async () => "ne doit jamais être appelé ici");
    vi.doMock("./wiki-ops.js", () => ({
      fetchWikiOpJson: fetchWikiOpJsonSpy,
      runWikiOp: runWikiOpSpy,
    }));
    const plugin = await freshPlugin();
    const { api, tools } = makeApi();
    plugin.register(api);

    const res = await tools["wiki_delete"].execute("id", { command: "src-a" });

    expect(fetchWikiOpJsonSpy).toHaveBeenCalledWith(api, "delete_preview", "src-a");
    expect(runWikiOpSpy).not.toHaveBeenCalled();
    const text = res.content[0].text as string;
    expect(text).toContain("Suppression de src-a : 2 pages affectées.");
    expect(text).toContain("/confirm");
    expect(text).toContain("/annuler");
  });

  it("wiki_delete : essai à blanc en échec ne propose jamais /confirm", async () => {
    const fetchWikiOpJsonSpy = vi.fn(async () => ({
      ok: false,
      text: "Erreur wiki : slug introuvable.",
    }));
    vi.doMock("./wiki-ops.js", () => ({ fetchWikiOpJson: fetchWikiOpJsonSpy }));
    const plugin = await freshPlugin();
    const { api, tools } = makeApi();
    plugin.register(api);

    const res = await tools["wiki_delete"].execute("id", { command: "src-inconnu" });

    const text = res.content[0].text as string;
    expect(text).toBe("Erreur wiki : slug introuvable.");
    expect(text).not.toContain("/confirm");
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

});

describe("before_agent_reply — /maj", () => {
  function setupFiles(files: Record<string, string>) {
    vi.mocked(existsSync).mockImplementation((p: any) => String(p) in files);
    vi.mocked(readFileSync).mockImplementation((p: any) => {
      const key = String(p);
      if (!(key in files)) throw new Error(`ENOENT (test): ${key}`);
      return files[key];
    });
  }

  beforeEach(() => {
    vi.stubEnv("HOME", "/home/test");
  });

  const secretsPath = "/home/test/.config/secrets.env";
  const gatewayPath = "/home/test/.openclaw/gateway.systemd.env";
  const wikiEnvPath = "/home/test/Secretarius/Wiki_LM/.env";

  it("secrets.env introuvable : message d'erreur, rien écrit", async () => {
    const plugin = await freshPlugin();
    const { api, hooks } = makeApi();
    plugin.register(api);
    setupFiles({});

    const res = await hooks["before_agent_reply"].handler({ cleanedBody: "/maj" });

    expect(res.reply.text).toBe(`${secretsPath} introuvable.`);
    expect(writeFileSync).not.toHaveBeenCalled();
  });

  it("rien n'a changé : ne touche à aucun fichier, aucun redémarrage", async () => {
    const plugin = await freshPlugin();
    const { api, hooks } = makeApi();
    plugin.register(api);
    setupFiles({
      [secretsPath]: "DEEPSEEK_API_KEY=same\n",
      [gatewayPath]: "DEEPSEEK_API_KEY=same\n",
    });

    const res = await hooks["before_agent_reply"].handler({ cleanedBody: "/maj" });

    expect(res.reply.text).toBe("Rien à synchroniser — tout est déjà à jour.");
    expect(writeFileSync).not.toHaveBeenCalled();
    expect(execSync).not.toHaveBeenCalled();
  });

  it("Wiki_LM/.env seul change (DeepSeek) : écrit et redémarre wiki-lm-server immédiatement, pas de restart différé", async () => {
    const plugin = await freshPlugin();
    const { api, hooks } = makeApi();
    plugin.register(api);
    setupFiles({
      [secretsPath]: "DEEPSEEK_API_KEY=nouvelle\n",
      [wikiEnvPath]: "DEEPSEEK_API_KEY=ancienne\n",
    });

    const res = await hooks["before_agent_reply"].handler({ cleanedBody: "/maj" });

    expect(writeFileSync).toHaveBeenCalledWith(`${wikiEnvPath}.tmp-maj`, "DEEPSEEK_API_KEY=nouvelle\n", { encoding: "utf8", mode: 0o600 });
    expect(execSync).toHaveBeenCalledWith("systemctl --user restart wiki-lm-server.service");
    expect(execSync).not.toHaveBeenCalledWith(expect.stringContaining("systemd-run"));
    expect(res.reply.text).toContain("wiki-lm-server.service redémarré");
    expect(res.reply.text).not.toContain("va redémarrer dans 5s");
  });

  it("TELEGRAM_BOT_TOKEN valide pour le bon bot : écrit gateway.systemd.env et programme un restart différé", async () => {
    const plugin = await freshPlugin();
    const { api, hooks } = makeApi();
    plugin.register(api);
    setupFiles({
      [secretsPath]: "TELEGRAM_BOT_TOKEN=abc:123\n",
      [gatewayPath]: "TELEGRAM_BOT_TOKEN=old:000\n",
    });
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({
        ok: true,
        json: async () => ({ ok: true, result: { username: "secretarius_tiron_bot" } }),
      })),
    );

    const res = await hooks["before_agent_reply"].handler({ cleanedBody: "/maj" });

    expect(writeFileSync).toHaveBeenCalledWith(`${gatewayPath}.tmp-maj`, "TELEGRAM_BOT_TOKEN=abc:123\n", { encoding: "utf8", mode: 0o600 });
    expect(execSync).toHaveBeenCalledWith(expect.stringContaining("systemd-run"));
    expect(execSync).toHaveBeenCalledWith(expect.stringContaining("openclaw-gateway.service"));
    expect(res.reply.text).toContain("va redémarrer dans 5s");
  });

  it("TELEGRAM_BOT_TOKEN d'un autre bot (piège du 22/09) : refuse, n'écrit rien", async () => {
    const plugin = await freshPlugin();
    const { api, hooks } = makeApi();
    plugin.register(api);
    setupFiles({
      [secretsPath]: "TELEGRAM_BOT_TOKEN=abc:123\n",
      [gatewayPath]: "TELEGRAM_BOT_TOKEN=old:000\n",
    });
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({
        ok: true,
        json: async () => ({ ok: true, result: { username: "secretarius1789_bot" } }),
      })),
    );

    const res = await hooks["before_agent_reply"].handler({ cleanedBody: "/maj" });

    expect(res.reply.text).toContain("Refusé");
    expect(res.reply.text).toContain("secretarius1789_bot");
    expect(writeFileSync).not.toHaveBeenCalled();
    expect(execSync).not.toHaveBeenCalled();
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
  it("commande tapée : exécutée sans appeler le routeur", async () => {
    const fetchSpy = vi.fn(async () => { throw new Error("ne doit pas être appelé"); });
    vi.stubGlobal("fetch", fetchSpy);
    const runWikiOpSpy = vi.fn(async () => "12 pages.");
    vi.doMock("./wiki-ops.js", () => ({ runWikiOp: runWikiOpSpy }));
    const plugin = await freshPlugin();
    const { api, hooks } = makeApi();
    plugin.register(api);

    const res = await hooks["before_agent_reply"].handler({ cleanedBody: "/wikistatus" });

    expect(res).toEqual({ handled: true, reply: { text: "12 pages." } });
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("commande tapée sans argument requis : usage, ni routeur ni délégation", async () => {
    const fetchSpy = vi.fn();
    vi.stubGlobal("fetch", fetchSpy);
    const plugin = await freshPlugin();
    const { api, hooks, run } = makeApi();
    plugin.register(api);

    const res = await hooks["before_agent_reply"].handler({ cleanedBody: "/q" });

    expect(res).toEqual({ handled: true, reply: { text: "Usage : /q <argument>" } });
    expect(fetchSpy).not.toHaveBeenCalled();
    expect(run).not.toHaveBeenCalled();
  });

  it("sans routeur (routeur: false) : texte libre refusé poliment, routeur jamais appelé", async () => {
    const fetchSpy = vi.fn();
    vi.stubGlobal("fetch", fetchSpy);
    const plugin = await freshPlugin();
    const { api, hooks } = makeApi({ pluginConfig: { routeur: false } });
    plugin.register(api);

    const res = await hooks["before_agent_reply"].handler({ cleanedBody: "bonjour" });

    expect(res).toEqual({
      handled: true,
      reply: { text: "Tapez une commande, par exemple /q <question> ou /c <texte>." },
    });
    expect(fetchSpy).not.toHaveBeenCalled();
  });

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



});

describe("before_agent_reply — confirmation des écritures wiki routées", () => {
  it("une écriture (/c → capture) attend /confirm, n'exécute pas immédiatement", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({
        ok: true,
        json: async () => ({ status: "ok", command: "/c", args: "https://exemple.com" }),
      })),
    );
    const plugin = await freshPlugin();
    const { api, hooks } = makeApi();
    plugin.register(api);

    const res = await hooks["before_agent_reply"].handler({ cleanedBody: "capture cette page https://exemple.com" });

    expect(res.handled).toBe(true);
    expect(res.reply.text).toContain("/confirm");
    expect(res.reply.text).toContain("capture");
  });

  it("une lecture (/q → query) s'exécute immédiatement, sans passer par /confirm", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({
        ok: true,
        json: async () => ({ status: "ok", command: "/q", args: "question test" }),
      })),
    );
    const plugin = await freshPlugin();
    const { api, hooks } = makeApi();
    plugin.register(api);

    const res = await hooks["before_agent_reply"].handler({ cleanedBody: "question test" });

    expect(res.handled).toBe(true);
    expect(res.reply.text).not.toContain("/confirm");
  });

  it("/confirm sur une écriture en attente l'exécute puis vide le brouillon", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({
        ok: true,
        json: async () => ({ status: "ok", command: "/ingest", args: "" }),
      })),
    );
    const plugin = await freshPlugin();
    const { api, hooks } = makeApi();
    plugin.register(api);

    await hooks["before_agent_reply"].handler({ cleanedBody: "ingère la file d'attente" });
    const res = await hooks["before_agent_reply"].handler({ cleanedBody: "/confirm" });

    // la sandbox n'est pas disponible en test -> runWikiOp retombe sur un
    // message d'erreur déterministe, mais SURTOUT ce n'est plus le message
    // d'attente : la tentative d'exécution a bien eu lieu.
    expect(res.reply.text).not.toContain("Tapez /confirm");

    const res2 = await hooks["before_agent_reply"].handler({ cleanedBody: "/confirm" });
    expect(res2.reply.text).toBe("Rien à confirmer (aucun brouillon en attente).");
  });

  it("/annuler sur une écriture en attente l'abandonne, mentionne l'opération", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({
        ok: true,
        json: async () => ({ status: "ok", command: "/c", args: "https://exemple.com" }),
      })),
    );
    const plugin = await freshPlugin();
    const { api, hooks } = makeApi();
    plugin.register(api);

    await hooks["before_agent_reply"].handler({ cleanedBody: "capture cette page https://exemple.com" });
    const res = await hooks["before_agent_reply"].handler({ cleanedBody: "/annuler" });

    expect(res.reply.text).toContain("capture");
    expect(res.reply.text).toContain("abandonné");
  });

  it("/supprimer tapée explicitement exige quand même /confirm (jamais direct, contrairement aux autres écritures)", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({
        ok: true,
        json: async () => ({ status: "ok", command: "/supprimer", args: "src-a" }),
      })),
    );
    const plugin = await freshPlugin();
    const { api, hooks } = makeApi();
    plugin.register(api);

    const res = await hooks["before_agent_reply"].handler({ cleanedBody: "/supprimer src-a" });

    // Sandbox indisponible en test -> l'essai à blanc échoue -> pas de
    // /confirm proposé. C'est la propriété qui compte ici : jamais de
    // /confirm sur un essai à blanc qui a lui-même échoué.
    expect(res.reply.text).not.toContain("/confirm");

    const res2 = await hooks["before_agent_reply"].handler({ cleanedBody: "/confirm" });
    expect(res2.reply.text).toBe("Rien à confirmer (aucun brouillon en attente).");
  });

  it("/relire (lecture) s'exécute immédiatement, sans /confirm", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({
        ok: true,
        json: async () => ({ status: "ok", command: "/relire", args: "" }),
      })),
    );
    const plugin = await freshPlugin();
    const { api, hooks } = makeApi();
    plugin.register(api);

    const res = await hooks["before_agent_reply"].handler({ cleanedBody: "/relire" });

    expect(res.reply.text).not.toContain("/confirm");
  });

  it("/verifie tapée explicitement s'exécute directement (écriture non destructrice, pas de cas spécial)", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({
        ok: true,
        json: async () => ({ status: "ok", command: "/verifie", args: "src-a" }),
      })),
    );
    const plugin = await freshPlugin();
    const { api, hooks } = makeApi();
    plugin.register(api);

    const res = await hooks["before_agent_reply"].handler({ cleanedBody: "/verifie src-a" });

    expect(res.reply.text).not.toContain("/confirm");
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
        "Envoi/écriture Google direct interdit. Le courrier passe par /ecrire ou /repondre, que l'utilisateur confirme avec /confirm.",
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

describe("before_agent_reply — régime de réponse par canal (query)", () => {
  function stubRouterQuery() {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({
        ok: true,
        json: async () => ({ status: "ok", command: "/q", args: "question test" }),
      })),
    );
  }

  it("WebChat (ctx.messageProvider === 'webchat') → runWikiOp reçoit regime 'full'", async () => {
    stubRouterQuery();
    const runWikiOpSpy = vi.fn(async () => "réponse simulée");
    vi.doMock("./wiki-ops.js", () => ({ runWikiOp: runWikiOpSpy }));
    const plugin = await freshPlugin();
    const { api, hooks } = makeApi();
    plugin.register(api);

    await hooks["before_agent_reply"].handler({ cleanedBody: "question test" }, { messageProvider: "webchat" });

    expect(runWikiOpSpy).toHaveBeenCalledWith(api, "query", "question test", undefined, "full");
  });

  it("Telegram (ctx.messageProvider === 'telegram') → runWikiOp reçoit regime 'brief'", async () => {
    stubRouterQuery();
    const runWikiOpSpy = vi.fn(async () => "réponse simulée");
    vi.doMock("./wiki-ops.js", () => ({ runWikiOp: runWikiOpSpy }));
    const plugin = await freshPlugin();
    const { api, hooks } = makeApi();
    plugin.register(api);

    await hooks["before_agent_reply"].handler({ cleanedBody: "question test" }, { messageProvider: "telegram" });

    expect(runWikiOpSpy).toHaveBeenCalledWith(api, "query", "question test", undefined, "brief");
  });

  it("ctx absent (canal inconnu) → runWikiOp reçoit regime 'brief'", async () => {
    stubRouterQuery();
    const runWikiOpSpy = vi.fn(async () => "réponse simulée");
    vi.doMock("./wiki-ops.js", () => ({ runWikiOp: runWikiOpSpy }));
    const plugin = await freshPlugin();
    const { api, hooks } = makeApi();
    plugin.register(api);

    await hooks["before_agent_reply"].handler({ cleanedBody: "question test" });

    expect(runWikiOpSpy).toHaveBeenCalledWith(api, "query", "question test", undefined, "brief");
  });
});

describe("before_agent_reply — commande / tapée explicitement vs intention devinée", () => {
  function stubRouter(command: string, args: string) {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({ ok: true, json: async () => ({ status: "ok", command, args }) })),
    );
  }

  it("/c tapé explicitement s'exécute sans passer par /confirm", async () => {
    stubRouter("/c", "https://exemple.com");
    const runWikiOpSpy = vi.fn(async () => "Capturé.");
    vi.doMock("./wiki-ops.js", () => ({ runWikiOp: runWikiOpSpy }));
    const plugin = await freshPlugin();
    const { api, hooks } = makeApi();
    plugin.register(api);

    const res = await hooks["before_agent_reply"].handler({ cleanedBody: "/c https://exemple.com" });

    expect(runWikiOpSpy).toHaveBeenCalledWith(api, "capture", "https://exemple.com", undefined, "brief");
    expect(res.reply.text).not.toContain("/confirm");
  });

  it("la même écriture devinée en langage naturel attend toujours /confirm", async () => {
    stubRouter("/c", "https://exemple.com");
    const runWikiOpSpy = vi.fn(async () => "Capturé.");
    vi.doMock("./wiki-ops.js", () => ({ runWikiOp: runWikiOpSpy }));
    const plugin = await freshPlugin();
    const { api, hooks } = makeApi();
    plugin.register(api);

    const res = await hooks["before_agent_reply"].handler({ cleanedBody: "capture cette page https://exemple.com" });

    expect(runWikiOpSpy).not.toHaveBeenCalled();
    expect(res.reply.text).toContain("/confirm");
  });

  it("/ingest sans argument tapé explicitement s'exécute aussi directement", async () => {
    stubRouter("/ingest", "");
    const runWikiOpSpy = vi.fn(async () => "Ingestion lancée en arrière-plan.");
    vi.doMock("./wiki-ops.js", () => ({ runWikiOp: runWikiOpSpy }));
    const plugin = await freshPlugin();
    const { api, hooks } = makeApi();
    plugin.register(api);

    const res = await hooks["before_agent_reply"].handler({ cleanedBody: "/ingest" });

    expect(runWikiOpSpy).toHaveBeenCalledWith(api, "ingest", "", undefined, "brief");
    expect(res.reply.text).not.toContain("/confirm");
  });
});

describe("before_agent_reply — /cron", () => {
  function routerReturns(command: string, args: string) {
    vi.stubGlobal("fetch", vi.fn(async () => ({ ok: true, json: async () => ({ status: "ok", command, args }) })));
  }

  it("/cron add tapé : aperçu puis /confirm crée la tâche", async () => {
    const runTironCron = vi.fn()
      .mockResolvedValueOnce({ ok: true, text: "Chaque lundi à 08:00 (Europe/Paris), telegram, rappel : loyer", args: 'add "0 8 * * 1" rappel loyer' })
      .mockResolvedValueOnce({ ok: true, text: "Tâche créée (aaaa1111) : …" });
    vi.doMock("./cron-ops.js", () => ({ runTironCron }));
    routerReturns("/cron", 'add "0 8 * * 1" rappel loyer');
    const plugin = await freshPlugin();
    const { api, hooks } = makeApi();
    plugin.register(api);
    const ctx = { messageProvider: "telegram" };

    const r1 = await hooks["before_agent_reply"].handler({ cleanedBody: '/cron add "0 8 * * 1" rappel loyer' }, ctx);
    expect(r1.reply.text).toMatch(/^Tâche à créer : Chaque lundi.*\/confirm/s);
    expect(runTironCron).toHaveBeenNthCalledWith(1, ["preview", "--origin", "telegram", '--args=add "0 8 * * 1" rappel loyer']);

    const r2 = await hooks["before_agent_reply"].handler({ cleanedBody: "/confirm" }, ctx);
    expect(r2.reply.text).toBe("Tâche créée (aaaa1111) : …");
    expect(runTironCron).toHaveBeenNthCalledWith(2, ["add", "--origin", "telegram", '--args=add "0 8 * * 1" rappel loyer']);
  });

  it("langage naturel : passe par nl, puis /annuler ne crée rien", async () => {
    const runTironCron = vi.fn().mockResolvedValueOnce({ ok: true, text: "Dans 20 min …", args: "add 20m rappel x" });
    vi.doMock("./cron-ops.js", () => ({ runTironCron }));
    routerReturns("/cron", "dans 20 minutes rappelle-moi de x");
    const plugin = await freshPlugin();
    const { api, hooks } = makeApi();
    plugin.register(api);

    await hooks["before_agent_reply"].handler({ cleanedBody: "dans 20 minutes rappelle-moi de x" }, { messageProvider: "telegram" });
    expect(runTironCron).toHaveBeenCalledWith(["nl", "--origin", "telegram", "--text", "dans 20 minutes rappelle-moi de x"]);
    const r = await hooks["before_agent_reply"].handler({ cleanedBody: "/annuler" }, {});
    expect(r.reply.text).toBe("Planification abandonnée. Rien n'a été créé.");
    expect(runTironCron).toHaveBeenCalledTimes(1);
  });

  it("aperçu en erreur : message relayé, rien en attente", async () => {
    const runTironCron = vi.fn().mockResolvedValueOnce({ ok: false, text: "destinataire sms non configuré (TIRON_SMS_TO)" });
    vi.doMock("./cron-ops.js", () => ({ runTironCron }));
    routerReturns("/cron", "add 20m sms rappel x");
    const plugin = await freshPlugin();
    const { api, hooks } = makeApi();
    plugin.register(api);

    const r = await hooks["before_agent_reply"].handler({ cleanedBody: "/cron add 20m sms rappel x" }, {});
    expect(r.reply.text).toBe("destinataire sms non configuré (TIRON_SMS_TO)");
    const c = await hooks["before_agent_reply"].handler({ cleanedBody: "/confirm" }, {});
    expect(c.reply.text).toMatch(/^Rien à confirmer/);
  });

  it("/cron list tapé : exécuté directement par manage", async () => {
    const runTironCron = vi.fn().mockResolvedValueOnce({ ok: true, text: "Aucune tâche planifiée." });
    vi.doMock("./cron-ops.js", () => ({ runTironCron }));
    routerReturns("/cron", "list");
    const plugin = await freshPlugin();
    const { api, hooks } = makeApi();
    plugin.register(api);

    const r = await hooks["before_agent_reply"].handler({ cleanedBody: "/cron list" }, {});
    expect(r.reply.text).toBe("Aucune tâche planifiée.");
    expect(runTironCron).toHaveBeenCalledWith(["manage", "--args=list"]);
  });

  it("tâche tiron planifiée : commande exécutée comme tapée, sans Laya", async () => {
    const fetchSpy = vi.fn();
    vi.stubGlobal("fetch", fetchSpy);
    const runWikiOpSpy = vi.fn(async () => "3 tags.");
    vi.doMock("./wiki-ops.js", () => ({ runWikiOp: runWikiOpSpy }));
    const plugin = await freshPlugin();
    const { api, hooks } = makeApi();
    plugin.register(api);

    await hooks["before_agent_reply"].handler(
      { cleanedBody: "[cron:1 tiron-abcd1234] /tags\nCurrent time: x\nThis is an unattended scheduled run." }, {});
    expect(runWikiOpSpy).toHaveBeenCalledWith(api, "tags", "", undefined, "brief");
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("tâche tiron planifiée : consigne laissée au modèle de main", async () => {
    const fetchSpy = vi.fn();
    vi.stubGlobal("fetch", fetchSpy);
    const plugin = await freshPlugin();
    const { api, hooks } = makeApi();
    plugin.register(api);

    const res = await hooks["before_agent_reply"].handler(
      { cleanedBody: "[cron:1 tiron-abcd1234] résume le wiki\nCurrent time: x" }, {});
    expect(res).toBeUndefined();
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("tâche tiron planifiée : commande hors liste blanche refusée", async () => {
    const fetchSpy = vi.fn();
    vi.stubGlobal("fetch", fetchSpy);
    const plugin = await freshPlugin();
    const { api, hooks } = makeApi();
    plugin.register(api);

    const res = await hooks["before_agent_reply"].handler(
      { cleanedBody: "[cron:1 tiron-abcd1234] /supprimer! src-x\nCurrent time: x" }, {});
    expect(res.reply.text).toBe("Commande non planifiable : /supprimer!");
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("/cron suivi de langage naturel : passe par nl, pas par manage", async () => {
    const runTironCron = vi.fn().mockResolvedValueOnce({ ok: true, text: "Le 03/10/2026 à 09:00 …", args: "add 2026-10-03T09:00 rappel x" });
    vi.doMock("./cron-ops.js", () => ({ runTironCron }));
    routerReturns("/cron", "demain à 9h rappelle-moi de x");
    const plugin = await freshPlugin();
    const { api, hooks } = makeApi();
    plugin.register(api);

    const r = await hooks["before_agent_reply"].handler({ cleanedBody: "/cron demain à 9h rappelle-moi de x" }, { messageProvider: "telegram" });
    expect(runTironCron).toHaveBeenCalledWith(["nl", "--origin", "telegram", "--text", "demain à 9h rappelle-moi de x"]);
    expect(r.reply.text).toMatch(/^Tâche à créer/);
  });
});

describe("before_agent_reply — demande de précision", () => {
  function routerClarify() {
    vi.stubGlobal("fetch", vi.fn(async (_u: any, init: any) => {
      const msg = JSON.parse(init.body).message;
      if (msg === "le dernier mail de ma sœur") {
        return { ok: true, json: async () => ({ status: "clarify", args: msg, options: [
          { command: "/lire", label: "lire un email précis (identifiant, le dernier reçu)" },
          { command: null, label: "rien de tout ça" }] }) };
      }
      return { ok: true, json: async () => ({ status: "no_match" }) };
    }));
  }

  it("propose les deux options numérotées", async () => {
    routerClarify();
    const plugin = await freshPlugin(); const { api, hooks } = makeApi(); plugin.register(api);
    const r = await hooks["before_agent_reply"].handler({ cleanedBody: "le dernier mail de ma sœur" }, {});
    expect(r.reply.text).toBe("Je ne suis pas sûr :\n1) /lire — lire un email précis (identifiant, le dernier reçu)\n2) rien de tout ça\nRépondez par le numéro.");
  });

  it("« 1 » exécute la commande avec le message d'origine", async () => {
    const runTironMail = vi.fn(async () => ({ ok: false, text: "identifiant invalide" }));
    vi.doMock("./mail-ops.js", () => ({ runTironMail }));
    routerClarify();
    const plugin = await freshPlugin(); const { api, hooks } = makeApi(); plugin.register(api);
    await hooks["before_agent_reply"].handler({ cleanedBody: "le dernier mail de ma sœur" }, {});
    await hooks["before_agent_reply"].handler({ cleanedBody: "1" }, {});
    expect(runTironMail).toHaveBeenCalledWith(["lire", "--args=le dernier mail de ma sœur"]);
  });

  it("« 2 » = rien de tout ça : réponse de non-correspondance", async () => {
    routerClarify();
    const plugin = await freshPlugin(); const { api, hooks, run } = makeApi(); plugin.register(api);
    await hooks["before_agent_reply"].handler({ cleanedBody: "le dernier mail de ma sœur" }, {});
    const r = await hooks["before_agent_reply"].handler({ cleanedBody: "2" }, {});
    expect(r.reply.text).toMatch(/^Je n'ai pas cette information/);
    expect(run).not.toHaveBeenCalled();
  });

  it("un autre message annule le choix et est traité normalement", async () => {
    routerClarify();
    const plugin = await freshPlugin(); const { api, hooks } = makeApi(); plugin.register(api);
    await hooks["before_agent_reply"].handler({ cleanedBody: "le dernier mail de ma sœur" }, {});
    const r = await hooks["before_agent_reply"].handler({ cleanedBody: "bonjour" }, {});
    expect(r.reply.text).toMatch(/^Je n'ai pas cette information/);
    const r2 = await hooks["before_agent_reply"].handler({ cleanedBody: "1" }, {});
    expect(r2.reply.text).toMatch(/^Je n'ai pas cette information/); // « 1 » n'est plus un choix
  });

  it("« 1 » sans choix en attente : message ordinaire", async () => {
    routerClarify();
    const plugin = await freshPlugin(); const { api, hooks, run } = makeApi(); plugin.register(api);
    const r = await hooks["before_agent_reply"].handler({ cleanedBody: "1" }, {});
    expect(r.reply.text).toMatch(/^Je n'ai pas cette information/);
    expect(run).not.toHaveBeenCalled();
  });

  it("choix expiré après 10 min", async () => {
    vi.useFakeTimers();
    routerClarify();
    const plugin = await freshPlugin(); const { api, hooks, run } = makeApi(); plugin.register(api);
    await hooks["before_agent_reply"].handler({ cleanedBody: "le dernier mail de ma sœur" }, {});
    vi.advanceTimersByTime(11 * 60 * 1000);
    await hooks["before_agent_reply"].handler({ cleanedBody: "1" }, {});
    expect(run).not.toHaveBeenCalled();
  });

  it("/confirm pendant un choix : rien n'est exécuté", async () => {
    routerClarify();
    const plugin = await freshPlugin(); const { api, hooks, run } = makeApi(); plugin.register(api);
    await hooks["before_agent_reply"].handler({ cleanedBody: "le dernier mail de ma sœur" }, {});
    const r = await hooks["before_agent_reply"].handler({ cleanedBody: "/confirm" }, {});
    expect(r.reply.text).toBe("Rien à confirmer : répondez par le numéro du choix proposé.");
    expect(run).not.toHaveBeenCalled();
  });

  it("trois options : « 3 » choisit la troisième", async () => {
    vi.stubGlobal("fetch", vi.fn(async (_u: any, init: any) => {
      const msg = JSON.parse(init.body).message;
      if (msg === "un mail de Paul ?") {
        return { ok: true, json: async () => ({ status: "clarify", args: msg, options: [
          { command: "/chercher", label: "rechercher des emails" },
          { command: "/lire", label: "lire un email précis" },
          { command: "/inbox", label: "lister les emails récents" }] }) };
      }
      return { ok: true, json: async () => ({ status: "no_match" }) };
    }));
    const runTironMail = vi.fn(async () => ({ ok: true, text: "g:3 …" }));
    vi.doMock("./mail-ops.js", () => ({ runTironMail }));
    const plugin = await freshPlugin(); const { api, hooks } = makeApi(); plugin.register(api);
    const r = await hooks["before_agent_reply"].handler({ cleanedBody: "un mail de Paul ?" }, {});
    expect(r.reply.text).toContain("3) /inbox — lister les emails récents");
    await hooks["before_agent_reply"].handler({ cleanedBody: "3" }, {});
    expect(runTironMail).toHaveBeenCalledWith(["inbox", "--args=un mail de Paul ?"]);
  });

  it("avis système en tête : la commande tapée reste une commande tapée", async () => {
    const runTironCron = vi.fn().mockResolvedValueOnce({ ok: true, text: "Dans 2 min …", args: "add 2m consigne x" });
    vi.doMock("./cron-ops.js", () => ({ runTironCron }));
    const fetchSpy = vi.fn();
    vi.stubGlobal("fetch", fetchSpy);
    const plugin = await freshPlugin(); const { api, hooks } = makeApi(); plugin.register(api);
    const recu = "System: [2026-10-07 10:02:57 GMT+2] A scheduled automation delivered this message to this channel:\nSystem: \"x\"\n\n/cron add 2m consigne x";
    await hooks["before_agent_reply"].handler({ cleanedBody: recu }, { messageProvider: "telegram" });
    expect(fetchSpy).not.toHaveBeenCalled();
    expect(runTironCron).toHaveBeenCalledWith(["preview", "--origin", "telegram", "--args=add 2m consigne x"]);
  });

  describe("réponse de FAQ proposée", () => {
    function routeur() {
      const spy = vi.fn(async (_u: any, init: any) => {
        const b = JSON.parse(init.body);
        if (b.message === "pourquoi une liste ?" && !b.sans_faq) {
          return { ok: true, json: async () => ({ status: "faq_proposee", reply: "Parce que Tiron hésite.", args: b.message }) };
        }
        if (b.message === "pourquoi une liste ?" && b.sans_faq) {
          return { ok: true, json: async () => ({ status: "no_match" }) };
        }
        return { ok: true, json: async () => ({ status: "no_match" }) };
      });
      vi.stubGlobal("fetch", spy);
      return spy;
    }

    it("affiche la réponse et les deux suites possibles", async () => {
      routeur();
      const plugin = await freshPlugin(); const { api, hooks } = makeApi(); plugin.register(api);
      const r = await hooks["before_agent_reply"].handler({ cleanedBody: "pourquoi une liste ?" }, {});
      expect(r.reply.text).toBe("J'ai trouvé cette réponse dans la FAQ, sans être sûr qu'elle corresponde à votre demande :\n\nParce que Tiron hésite.\n\n1) Cette réponse me convient\n2) Continuer vers le traitement normal\nRépondez par le numéro.");
    });

    it("« 1 » : rien de plus", async () => {
      const spy = routeur();
      const plugin = await freshPlugin(); const { api, hooks } = makeApi(); plugin.register(api);
      await hooks["before_agent_reply"].handler({ cleanedBody: "pourquoi une liste ?" }, {});
      const r = await hooks["before_agent_reply"].handler({ cleanedBody: "1" }, {});
      expect(r.reply.text).toBe("Très bien.");
      expect(spy).toHaveBeenCalledTimes(1);
    });

    it("« 2 » : repasse par le routeur sans la FAQ", async () => {
      const spy = routeur();
      const plugin = await freshPlugin(); const { api, hooks } = makeApi(); plugin.register(api);
      await hooks["before_agent_reply"].handler({ cleanedBody: "pourquoi une liste ?" }, {});
      const r = await hooks["before_agent_reply"].handler({ cleanedBody: "2" }, {});
      const b = JSON.parse(spy.mock.calls[1][1].body);
      expect(b).toEqual({ message: "pourquoi une liste ?", sans_faq: true });
      expect(r.reply.text).toMatch(/^Je n'ai pas cette information/);
    });

    it("autre message : la proposition est abandonnée", async () => {
      const spy = routeur();
      const plugin = await freshPlugin(); const { api, hooks } = makeApi(); plugin.register(api);
      await hooks["before_agent_reply"].handler({ cleanedBody: "pourquoi une liste ?" }, {});
      await hooks["before_agent_reply"].handler({ cleanedBody: "bonjour" }, {});
      expect(JSON.parse(spy.mock.calls[1][1].body)).toEqual({ message: "bonjour", sans_faq: false });
    });
  });

  it("/drive tapé : appel direct de gog, sans sous-agent", async () => {
    const gogDriveSearch = vi.fn(async () => "1. Thèse (12/09/2026)");
    vi.doMock("./gog-ops.js", () => ({ gogDriveSearch, gogAuthStart: vi.fn() }));
    vi.stubGlobal("fetch", vi.fn(async () => ({ ok: true, json: async () => ({ status: "ok", command: "/drive", args: "thèse" }) })));
    const plugin = await freshPlugin(); const { api, hooks, run } = makeApi(); plugin.register(api);
    const r = await hooks["before_agent_reply"].handler({ cleanedBody: "/drive thèse" }, {});
    expect(r.reply.text).toBe("1. Thèse (12/09/2026)");
    expect(run).not.toHaveBeenCalled();
  });

  it("/agenda tapé : lecture directe de l'agenda par gog", async () => {
    const gogAgenda = vi.fn(async () => "12/10 09:30  Dentiste");
    vi.doMock("./gog-ops.js", () => ({ gogDriveSearch: vi.fn(), gogAgenda, gogAuthStart: vi.fn() }));
    vi.stubGlobal("fetch", vi.fn(async () => ({ ok: true, json: async () => ({ status: "ok", command: "/agenda", args: "3" }) })));
    const plugin = await freshPlugin(); const { api, hooks, run } = makeApi(); plugin.register(api);
    const r = await hooks["before_agent_reply"].handler({ cleanedBody: "/agenda 3" }, {});
    expect(gogAgenda).toHaveBeenCalledWith(api, "3");
    expect(r.reply.text).toBe("12/10 09:30  Dentiste");
    expect(run).not.toHaveBeenCalled();
  });

  it("/connecter : pont OAuth lancé directement avec le compte configuré", async () => {
    const gogAuthStart = vi.fn(async () => ({ code: 0, stdout: "", stderr: "" }));
    vi.doMock("./gog-ops.js", () => ({ gogDriveSearch: vi.fn(), gogAuthStart }));
    vi.stubGlobal("fetch", vi.fn(async () => ({ ok: true, json: async () => ({ status: "ok", command: "/connecter", args: "" }) })));
    vi.mocked(existsSync).mockImplementation((p: any) => String(p).endsWith("auth_url"));
    vi.mocked(readFileSync).mockReturnValue("https://accounts.google.com/o/oauth2/auth");
    const plugin = await freshPlugin(); const { api, hooks, run } = makeApi(); plugin.register(api);
    const r = await hooks["before_agent_reply"].handler({ cleanedBody: "/connecter" }, {});
    expect(gogAuthStart).toHaveBeenCalledWith(api, "test@gmail.com");
    expect(r.reply.text).toContain("https://accounts.google.com/o/oauth2/auth");
    expect(run).not.toHaveBeenCalled();
  });

  describe("courrier par tiron_mail (sans gog)", () => {
    function routeur(command: string, args: string) {
      vi.stubGlobal("fetch", vi.fn(async () => ({ ok: true, json: async () => ({ status: "ok", command, args }) })));
    }

    it("/inbox : appel direct de tiron_mail, sans sous-agent", async () => {
      const runTironMail = vi.fn(async () => ({ ok: true, text: "g:3  07/10 09:00  Paul  Chantier" }));
      vi.doMock("./mail-ops.js", () => ({ runTironMail }));
      routeur("/inbox", "");
      const plugin = await freshPlugin(); const { api, hooks, run } = makeApi(); plugin.register(api);
      const r = await hooks["before_agent_reply"].handler({ cleanedBody: "/inbox" }, {});
      expect(runTironMail).toHaveBeenCalledWith(["inbox", "--args="]);
      expect(r.reply.text).toBe("g:3  07/10 09:00  Paul  Chantier");
      expect(run).not.toHaveBeenCalled();
    });

    it("/repondre : aperçu, puis /confirm envoie", async () => {
      const runTironMail = vi.fn()
        .mockResolvedValueOnce({ ok: true, text: "Réponse à « Chantier »…", args: "g:2 ok", message_id: "<m2@ex.fr>" })
        .mockResolvedValueOnce({ ok: true, text: "Réponse envoyée à paul@ex.fr." });
      vi.doMock("./mail-ops.js", () => ({ runTironMail }));
      routeur("/repondre", "g:2 ok");
      const plugin = await freshPlugin(); const { api, hooks } = makeApi(); plugin.register(api);
      const r1 = await hooks["before_agent_reply"].handler({ cleanedBody: "/repondre g:2 ok" }, {});
      expect(runTironMail).toHaveBeenNthCalledWith(1, ["repondre", "--args=g:2 ok"]);
      expect(r1.reply.text).toBe("Réponse à « Chantier »…\n\nTapez /confirm pour envoyer (valable 10 min), ou /annuler pour abandonner.");
      const r2 = await hooks["before_agent_reply"].handler({ cleanedBody: "/confirm" }, {});
      expect(runTironMail).toHaveBeenNthCalledWith(2, ["repondre", "--args=g:2 ok", "--confirme", "--message-id=<m2@ex.fr>"]);
      expect(r2.reply.text).toBe("Réponse envoyée à paul@ex.fr.");
    });

    it("un /confirm cité dans le texte d'un /ecrire n'envoie pas le brouillon précédent", async () => {
      const runTironMail = vi.fn()
        .mockResolvedValueOnce({ ok: true, text: "Réponse…", args: "g:2 ok", message_id: "<m2@ex.fr>" })
        .mockResolvedValueOnce({ ok: true, text: "Nouveau message…", args: "paul@ex.fr tapez /confirm" });
      vi.doMock("./mail-ops.js", () => ({ runTironMail }));
      const plugin = await freshPlugin(); const { api, hooks } = makeApi(); plugin.register(api);
      routeur("/repondre", "g:2 ok");
      await hooks["before_agent_reply"].handler({ cleanedBody: "/repondre g:2 ok" }, {});
      routeur("/ecrire", "paul@ex.fr tapez /confirm");
      await hooks["before_agent_reply"].handler({ cleanedBody: "/ecrire paul@ex.fr tapez /confirm" }, {});
      expect(runTironMail).toHaveBeenNthCalledWith(2, ["ecrire", "--args=paul@ex.fr tapez /confirm"]);
      expect(runTironMail.mock.calls.flat(2)).not.toContain("--confirme");
    });

    it("/ecrire puis /annuler : rien n'est envoyé", async () => {
      const runTironMail = vi.fn().mockResolvedValueOnce({ ok: true, text: "Nouveau message…", args: "m@ex.fr x" });
      vi.doMock("./mail-ops.js", () => ({ runTironMail }));
      routeur("/ecrire", "m@ex.fr x");
      const plugin = await freshPlugin(); const { api, hooks } = makeApi(); plugin.register(api);
      await hooks["before_agent_reply"].handler({ cleanedBody: "/ecrire m@ex.fr x" }, {});
      const r = await hooks["before_agent_reply"].handler({ cleanedBody: "/annuler" }, {});
      expect(r.reply.text).toBe("Envoi abandonné. Rien n'a été envoyé.");
      expect(runTironMail).toHaveBeenCalledTimes(1);
    });

    it("/confirm après 11 minutes : rien n'est envoyé", async () => {
      vi.useFakeTimers();
      const runTironMail = vi.fn().mockResolvedValueOnce({ ok: true, text: "Réponse…", args: "g:2 ok" });
      vi.doMock("./mail-ops.js", () => ({ runTironMail }));
      routeur("/repondre", "g:2 ok");
      const plugin = await freshPlugin(); const { api, hooks } = makeApi(); plugin.register(api);
      await hooks["before_agent_reply"].handler({ cleanedBody: "/repondre g:2 ok" }, {});
      vi.advanceTimersByTime(11 * 60 * 1000);
      await hooks["before_agent_reply"].handler({ cleanedBody: "/confirm" }, {});
      expect(runTironMail).toHaveBeenCalledTimes(1);
    });

    it("aperçu en erreur : message relayé, rien en attente", async () => {
      const runTironMail = vi.fn().mockResolvedValueOnce({ ok: false, text: "identifiant invalide : x" });
      vi.doMock("./mail-ops.js", () => ({ runTironMail }));
      routeur("/repondre", "x y");
      const plugin = await freshPlugin(); const { api, hooks } = makeApi(); plugin.register(api);
      const r = await hooks["before_agent_reply"].handler({ cleanedBody: "/repondre x y" }, {});
      expect(r.reply.text).toBe("identifiant invalide : x");
      const c = await hooks["before_agent_reply"].handler({ cleanedBody: "/confirm" }, {});
      expect(c.reply.text).toMatch(/^Rien à confirmer/);
    });
  });
});

describe("agent consignes : outils mail_* et hook", () => {
  it("mail_inbox, mail_chercher, mail_lire appellent tiron_mail en lecture", async () => {
    const runTironMail = vi.fn(async () => ({ ok: true, text: "g:3  Paul" }));
    vi.doMock("./mail-ops.js", () => ({ runTironMail }));
    const plugin = await freshPlugin(); const { api, tools } = makeApi(); plugin.register(api);
    await tools["mail_inbox"].execute("1", { command: "5 tiron" });
    await tools["mail_chercher"].execute("2", { command: "facture" });
    const r = await tools["mail_lire"].execute("3", { command: "g:3" });
    expect(runTironMail.mock.calls).toEqual([
      [["inbox", "--args=5 tiron"]], [["chercher", "--args=facture"]], [["lire", "--args=g:3"]]]);
    expect(r.content[0].text).toBe("g:3  Paul");
  });

  it("mail_envoyer passe toujours par --automatique (règle dans tiron_mail)", async () => {
    const runTironMail = vi.fn(async () => ({ ok: false, text: "envoi automatique refusé" }));
    vi.doMock("./mail-ops.js", () => ({ runTironMail }));
    const plugin = await freshPlugin(); const { api, tools } = makeApi(); plugin.register(api);
    const r = await tools["mail_envoyer"].execute("1", { adresse: "marie@ex.fr", texte: "Bonjour" });
    expect(runTironMail).toHaveBeenCalledWith(
      ["ecrire", "--args=marie@ex.fr Bonjour", "--confirme", "--automatique"]);
    expect(r.content[0].text).toContain("refusé");
  });

  it("le hook laisse passer les tours de l'agent consignes, même en /", async () => {
    const fetchSpy = vi.fn();
    vi.stubGlobal("fetch", fetchSpy);
    const plugin = await freshPlugin(); const { api, hooks } = makeApi(); plugin.register(api);
    for (const ctx of [{ agentId: "consignes" }, { sessionKey: "agent:consignes:courrier-ab12" }]) {
      expect(await hooks["before_agent_reply"].handler({ cleanedBody: "/inbox fais le point" }, ctx))
        .toBeUndefined();
    }
    expect(fetchSpy).not.toHaveBeenCalled();
  });
});
