// /jeton : jetons des agents de la porte Grok (spec 2026-10-13-porte-grok),
// par appel direct de porte-jeton sur l'hôte. Telegram seulement : la réponse
// ne doit jamais atterrir dans une note du coffre, lisible par les agents.
import { existsSync } from "node:fs";
import { spawnSync } from "node:child_process";

const USAGE = "Usage : /jeton ajouter <préfixe> | /jeton retirer <préfixe> | /jeton lister";

export function commandeJeton(arg: string, bin: string): string {
  const mots = arg.trim().split(/\s+/).filter(Boolean);
  const [action, ...reste] = mots;
  const valide =
    ((action === "ajouter" || action === "retirer") && reste.length === 1) ||
    (action === "lister" && reste.length === 0);
  if (!valide) return USAGE;
  if (!existsSync(bin)) return "Porte absente sur cette machine.";

  // Sans shell : l'argument est passé tel quel, jamais interprété.
  const r = spawnSync(bin, mots, { encoding: "utf8", timeout: 30000 });
  if (r.status !== 0) return (r.stderr || r.error?.message || "échec de porte-jeton").trim();
  const sortie = r.stdout.trim();
  return sortie || (action === "lister" ? "Aucun agent." : "");
}
