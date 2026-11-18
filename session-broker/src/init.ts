import { chmodSync, existsSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { randomBytes } from "node:crypto";

// Crée l'identité Session de Tiron : la phrase secrète ne quitte jamais
// secrets.env ; seule l'identité publique est rendue.
export function initIdentity(
  secretsPath: string,
  gen: () => string,
  sessionIdFrom: (mnemonic: string) => string,
): { sessionId: string } {
  const existe = existsSync(secretsPath);
  const avant = existe ? readFileSync(secretsPath, "utf8") : "";
  if (/^SESSION_BOT_MNEMONIC=/m.test(avant)) throw new Error("SESSION_BOT_MNEMONIC existe déjà dans secrets.env : rien n'est modifié");
  const mode = existe ? statSync(secretsPath).mode & 0o777 : 0o600;
  const mnemonic = gen();
  const sessionId = sessionIdFrom(mnemonic);
  const sep = avant && !avant.endsWith("\n") ? "\n" : "";
  const ajout = `SESSION_BOT_MNEMONIC=${mnemonic}\nTIRON_SESSION_BROKER_TOKEN=${randomBytes(32).toString("hex")}\n`;
  writeFileSync(secretsPath, avant + sep + ajout, { mode });
  chmodSync(secretsPath, mode);
  return { sessionId };
}
