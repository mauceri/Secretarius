export const SUPPORTED_COMMANDS = [
  "/c",
  "/q",
  "/ingest",
  "/wikistatus",
  "/r",
  "/tags",
  "/kbupdate",
  "/relire",
  "/verifie",
  "/supprimer?",
  "/supprimer!",
  "/lint",
  "/repair?",
  "/repair!",
  "/switch-wiki-model",
  "/cron?",
  "/cron!",
  "/cron",
  "/mail?",
  "/mail!",
  "/help",
] as const;

export interface WikiBlockMatch {
  command: string;
  arg: string;
  content: string;
  previousResult?: string;
  start: number;
  end: number;
}

const BLOCK_RE =
  /```wiki\r?\n([\s\S]*?)\r?\n```(?:\r?\n+<!-- wikilm-run:start -->\r?\n?([\s\S]*?)\r?\n?<!-- wikilm-run:end -->)?/g;

// Note de veille (porte Grok, 2026-10-13) : une case vide « - [ ] » sur la
// dernière ligne non vide avant un bloc l'écarte ; cochée, ou absente, le
// bloc s'exécute comme avant.
const CASE_VIDE = /^\s*[-*]\s+\[ \]/;

function precedeDUneCaseVide(noteText: string, debut: number): boolean {
  const lignes = noteText.slice(0, debut).split("\n").filter((l) => l.trim() !== "");
  return lignes.length > 0 && CASE_VIDE.test(lignes[lignes.length - 1]);
}

export function extractWikiBlocks(noteText: string): WikiBlockMatch[] {
  const blocks: WikiBlockMatch[] = [];
  const re = new RegExp(BLOCK_RE.source, "g");
  let match: RegExpExecArray | null;
  while ((match = re.exec(noteText)) !== null) {
    if (precedeDUneCaseVide(noteText, match.index)) continue;
    const content = match[1];
    const lines = content.split("\n");
    const command = lines[0].trim();
    const arg = lines.slice(1).join("\n").trim();
    blocks.push({
      command,
      arg,
      content,
      ...(match[2] !== undefined ? { previousResult: match[2] } : {}),
      start: match.index,
      end: match.index + match[0].length,
    });
  }
  return blocks;
}

const ENTETE_MAIL = /^\s*(à|a|objet)\s*:/i;

const ENTETE_A = /^\s*(à|a)\s*:/im;

const PREFIXE_USAGE = "Argument manquant. Écrivez-le sur la ligne qui suit la commande, dans le bloc :\n\n";

const CRON_ADD = `<quand> [<canal>] <type> <contenu>   (sur la ligne suivante, précédé de « add »)
  quand   : "0 8 * * 1" (format cron, entre guillemets), 2026-11-05T09:00, ou 20m / 2h / 1d.
  canal   : facultatif — telegram, whatsapp, sms, obsidian (par défaut : obsidian).
  type    : rappel (texte envoyé), commande (/q, /r, /tags…), consigne (tâche pour l'agent).
  contenu : le texte, la commande ou la consigne.`;

// Blocs dont l'argument est obligatoire, avec le texte écrit dans la zone de
// résultat quand il manque. /repair et /switch-wiki-model ont déjà une
// réponse utile sans argument ; /mail est traité à part (ligne « à : »).
const USAGES_BLOC: Record<string, string> = {
  "/c": `/c <texte ou URL>
  texte ou URL : la note à capturer, ou l'adresse d'une page web.
  Options dans le texte :
    #tag      étiquette la capture
    ref:<url> cite une source
    @simple   écrit la note telle quelle comme page source du wiki, sans ingestion
    file:<chemin> ajoute le contenu d'un fichier déjà déposé dans le wiki (pièce jointe)`,
  "/q": `/q <question>
  question : en langage courant. Réponse rédigée à partir du wiki (environ 1 min).`,
  "/r": `/r <mots-clés>
  mots-clés : recherche rapide sans synthèse, 5 résultats.`,
  "/verifie": `/verifie <page>
  page : nom de fichier de la page sans .md (ex. c-domestication-des-chiens).`,
  "/supprimer?": `/supprimer? <page>
  page : nom de fichier de la page sans .md. Essai à blanc : rien n'est supprimé.`,
  "/supprimer!": `/supprimer! <page>
  page : nom de fichier de la page sans .md. Suppression réelle, en cascade (vers la poubelle).`,
  "/cron?": `/cron? ${CRON_ADD}
  /cron? montre l'aperçu, rien n'est créé.`,
  "/cron!": `/cron! ${CRON_ADD}
  /cron! crée la tâche.`,
  "/cron": `/cron list | rm <id> | pause <id> | reprendre <id>
  Pour créer une tâche : /cron? (aperçu) ou /cron! (création).`,
};

const USAGE_MAIL = `/mail? ou /mail!
  à :     adresse du destinataire (ligne « à : … »).
  objet : facultatif (ligne « objet : … ») ; sinon la première ligne du corps.
  corps : les lignes suivantes ; si le bloc n'en a pas, la note elle-même.
  /mail? montre l'aperçu, /mail! envoie depuis tiron@secretarius.co.`;

// Texte d'usage si le bloc n'a pas l'argument qu'exige sa commande, sinon null.
export function usageBloc(block: WikiBlockMatch): string | null {
  if (block.command === "/mail?" || block.command === "/mail!") {
    return ENTETE_A.test(block.arg) ? null : PREFIXE_USAGE + USAGE_MAIL;
  }
  const usage = USAGES_BLOC[block.command];
  return usage && !block.arg ? PREFIXE_USAGE + usage : null;
}

// Bloc /mail? ou /mail! sans corps (seulement « à : » et « objet : ») : la
// note, sans ses blocs ```wiki ni leurs résultats, devient le corps (spec courrier).
export function blocArgMail(arg: string, noteText: string): string {
  const corps = arg.split("\n").filter((l) => l.trim() && !ENTETE_MAIL.test(l));
  if (corps.length > 0) return arg;
  const note = noteText
    .replace(/^---\r?\n[\s\S]*?\r?\n---\r?\n/, "") // propriétés de la note : jamais au destinataire
    .replace(new RegExp(BLOCK_RE.source, "g"), "")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
  return `${arg}\n${note}`;
}

// Relancer les commandes d'une note ne renvoie jamais un courriel déjà parti :
// seul un résultat « Courriel envoyé » bloque (un aperçu /mail? devenu /mail! part).
export function dejaEnvoye(block: WikiBlockMatch): boolean {
  return block.command === "/mail!" && (block.previousResult ?? "").startsWith("Courriel envoyé");
}

export function applyWikiResults(
  noteText: string,
  blocks: WikiBlockMatch[],
  resultTexts: string[]
): string {
  let output = "";
  let cursor = 0;
  blocks.forEach((block, i) => {
    output += noteText.slice(cursor, block.start);
    output += "```wiki\n" + block.content + "\n```\n\n";
    output += "<!-- wikilm-run:start -->\n" + resultTexts[i] + "\n<!-- wikilm-run:end -->";
    cursor = block.end;
  });
  output += noteText.slice(cursor);
  return output;
}

export const ATTENTE = "Évaluation en cours…";
export const BLOC_MODIFIE = "Bloc modifié pendant l'évaluation : relancez les commandes.";

// Fin de lot : la note a pu changer pendant l'attente. On ne remplace que les
// messages d'attente, chacun par le résultat du bloc de même contenu ; un bloc
// modifié entre-temps n'a plus de résultat qui lui corresponde.
export function finaliserLot(
  noteText: string,
  resultats: { content: string; result: string }[]
): string {
  const restants = [...resultats];
  const blocs: WikiBlockMatch[] = [];
  const textes: string[] = [];
  for (const block of extractWikiBlocks(noteText)) {
    if (block.previousResult?.trim() !== ATTENTE) continue;
    const i = restants.findIndex((r) => r.content === block.content);
    blocs.push(block);
    textes.push(i >= 0 ? restants.splice(i, 1)[0].result : BLOC_MODIFIE);
  }
  return applyWikiResults(noteText, blocs, textes);
}

export function formatWikiResult(
  command: string,
  data: Record<string, unknown>,
  ok: boolean
): string {
  if (!ok || typeof data.error === "string") {
    const message = typeof data.error === "string" ? data.error : "Erreur inconnue";
    return `**Erreur :** ${message}`;
  }
  switch (command) {
    case "/c": {
      const files = (data.files as string[] | undefined) ?? [];
      return files.length > 0 ? `Capturé : ${files.join(", ")}` : "Rien à capturer.";
    }
    case "/q":
      return String(data.synthesis ?? "");
    case "/ingest": {
      const status = data.status as string | undefined;
      if (status === "launched") return `Ingestion lancée (${data.queued} en attente).`;
      if (status === "already_running") return "Ingestion déjà en cours.";
      return "Rien à ingérer.";
    }
    case "/wikistatus": {
      const running = data.running ? "oui" : "non";
      const blocked = ((data.blocked_files as string[] | undefined) ?? []).length;
      return `En cours : ${running}. En attente : ${data.pending}. Bloqués : ${blocked}.`;
    }
    case "/r": {
      const results = (data.results as { title: string; excerpt: string }[] | undefined) ?? [];
      if (results.length === 0) return "Aucun résultat.";
      return results.map((r, i) => `${i + 1}. ${r.title} — ${r.excerpt}`).join("\n");
    }
    case "/tags": {
      const tags = (data.tags as string[] | undefined) ?? [];
      return tags.length > 0 ? tags.join(", ") : "Aucun tag.";
    }
    case "/kbupdate":
      if (data.status === "error") return `**Erreur :** ${data.reason ?? "échec de la mise à jour"}`;
      return `Base de connaissances mise à jour (clustering ${data.clustering}).`;
    case "/relire":
      return data.status === "empty" ? "Rien à relire." : String(data.content ?? "");
    case "/verifie":
      return `Page ${data.slug} marquée vérifiée.`;
    case "/supprimer?": {
      const affected = (data.affected as string[] | undefined) ?? [];
      return `Essai à blanc — ${affected.length} page(s) seraient supprimées : ${affected.join(", ")}.\n\nRien n'a été supprimé. Remplacez par /supprimer! pour confirmer.`;
    }
    case "/supprimer!": {
      const affected = (data.affected as string[] | undefined) ?? [];
      return `Supprimé : ${affected.length} page(s) — ${affected.join(", ")}.`;
    }
    case "/lint": {
      const byCode = (data.by_code as Record<string, number> | undefined) ?? {};
      const details = Object.entries(byCode).map(([code, n]) => `${code}: ${n}`).join(", ");
      return `${data.checked_pages} page(s) vérifiée(s) — ${data.errors} erreur(s), ${data.warnings} avertissement(s).${details ? `\n${details}` : ""}`;
    }
    case "/repair?":
    case "/repair!": {
      if (data.status === "families") {
        const available = (data.available as string[] | undefined) ?? [];
        return `Famille manquante. Familles disponibles : ${available.join(", ")}.`;
      }
      const changes = (data.changes as string[] | undefined) ?? [];
      const detail = changes.length > 0 ? `\n\n${changes.join("\n")}` : "";
      if (command === "/repair?") {
        return `Essai à blanc (${data.family}) — ${data.before_count} → ${data.after_count} après réparation.${detail}\n\nRien n'a été modifié. Remplacez par /repair! pour confirmer.`;
      }
      return `Réparé (${data.family}) — ${data.before_count} → ${data.after_count}.${detail}`;
    }
    case "/switch-wiki-model": {
      if (data.status === "current") {
        const available = (data.available as string[] | undefined) ?? [];
        const alias = data.alias ? String(data.alias) : "aucun alias connu";
        return `Modèle actuel : ${alias} — ${data.backend}/${data.model} (${data.base_url}).\nAlias disponibles : ${available.join(", ")}.`;
      }
      return `Modèle du wiki basculé sur ${data.alias} (${data.model}).`;
    }
    case "/cron?":
      return `${String(data.text ?? "")}\n\nRien n'a été créé. Remplacez par /cron! pour créer la tâche.`;
    case "/cron!":
    case "/cron":
      return String(data.text ?? "");
    case "/mail?":
      return `${String(data.text ?? "")}\n\nRien n'a été envoyé. Remplacez par /mail! pour envoyer.`;
    case "/mail!":
      return String(data.text ?? "");
    case "/help":
      return String(data.help ?? "");
    default:
      return `Commande non prise en charge en exécution par lot : ${command}`;
  }
}
