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
] as const;

export interface WikiBlockMatch {
  command: string;
  arg: string;
  content: string;
  start: number;
  end: number;
}

const BLOCK_RE =
  /```wiki\r?\n([\s\S]*?)\r?\n```(?:\r?\n+<!-- wikilm-run:start -->[\s\S]*?<!-- wikilm-run:end -->)?/g;

export function extractWikiBlocks(noteText: string): WikiBlockMatch[] {
  const blocks: WikiBlockMatch[] = [];
  const re = new RegExp(BLOCK_RE.source, "g");
  let match: RegExpExecArray | null;
  while ((match = re.exec(noteText)) !== null) {
    const content = match[1];
    const lines = content.split("\n");
    const command = lines[0].trim();
    const arg = lines.slice(1).join("\n").trim();
    blocks.push({
      command,
      arg,
      content,
      start: match.index,
      end: match.index + match[0].length,
    });
  }
  return blocks;
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
    default:
      return `Commande non prise en charge en exécution par lot : ${command}`;
  }
}
