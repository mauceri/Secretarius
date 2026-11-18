// Point d'entrée : `init` (création de l'identité) ou `serve` (démon).
import { Session, Poller, ready } from "@session.js/client";
import { generateSeedHex } from "@session.js/keypair";
import { encode } from "@session.js/mnemonic";
import { join } from "node:path";
import { loadConfig } from "./config";
import { createServer, type BrokerStatus } from "./http";
import { handleSessionMessage } from "./inbound";
import { initIdentity } from "./init";
import { Store } from "./store";

const HOME = process.env.HOME ?? "";
const log = (l: string) => console.log(`${new Date().toISOString()} ${l}`);
const POLL_MS = 3000;

async function init() {
  await ready;
  const { sessionId } = initIdentity(
    join(HOME, ".config", "secrets.env"),
    () => encode(generateSeedHex()),
    (m) => { const s = new Session(); s.setMnemonic(m); return s.getSessionID(); },
  );
  console.log(sessionId);
}

async function serve() {
  const cfg = loadConfig(join(HOME, ".config", "session-broker", "config.json"), process.env);
  const mnemonic = process.env.SESSION_BOT_MNEMONIC;
  if (!mnemonic) throw new Error("SESSION_BOT_MNEMONIC absent");
  delete process.env.SESSION_BOT_MNEMONIC;
  await ready;
  const session = new Session();
  session.setMnemonic(mnemonic, "Tiron");
  const store = new Store(join(cfg.stateDir, "broker.sqlite"));
  const status: BrokerStatus = { lastPollOkAt: null, lastSendOkAt: null, sessionId: session.getSessionID() };
  const app = createServer({ cfg, status, store, session: session as any, log });
  const deps = { session: session as any, store, cfg, publish: app.publish, log };
  session.on("message", (m: any) => {
    handleSessionMessage(m, deps).catch((e) => log(`erreur de réception : ${e instanceof Error ? e.message : String(e)}`));
  });
  // Poll manuel : permet de savoir quand le réseau a répondu pour la dernière fois.
  const poller = new Poller({ interval: null });
  session.addPoller(poller);
  let enCours = false;
  setInterval(async () => {
    if (enCours) return;
    enCours = true;
    try { await poller.poll(); status.lastPollOkAt = new Date().toISOString(); }
    catch (e) { log(`échec du poll : ${e instanceof Error ? e.message : String(e)}`); }
    finally { enCours = false; }
  }, POLL_MS);
  // idleTimeout 0 : le flux SSE reste ouvert (défaut Bun : 10 s).
  Bun.serve({ hostname: "127.0.0.1", port: cfg.port, fetch: app.fetch, idleTimeout: 0 });
  log(`session-broker prêt sur 127.0.0.1:${cfg.port}, identité ${status.sessionId.slice(0, 8)}`);
}

const cmd = process.argv[2];
if (cmd === "init") await init();
else if (cmd === "serve") await serve();
else { console.error("usage : main.ts init | serve"); process.exit(2); }
