// Formatage déterministe du JSON de wiki.py en message utilisateur.
// Aucune invention : sur erreur, on surface le texte de wiki.py verbatim.
export function formatWikiResult(op: string, json: any): string {
  if (json && typeof json.error === "string") return json.error;
  if (json && json.status === "error") return json.reason ?? json.error ?? "Erreur wiki.";

  switch (op) {
    case "query":
      return typeof json?.synthesis === "string" && json.synthesis.trim()
        ? json.synthesis
        : "Réponse wiki vide ou inattendue.";
    case "capture": {
      const files = Array.isArray(json?.files) ? json.files : [];
      return files.length
        ? `Capturé : ${files.join(", ")} (en file d'attente pour ingestion).`
        : "Rien à capturer.";
    }
    case "ingest":
      if (json?.status === "launched") return "Ingestion lancée en arrière-plan.";
      if (json?.status === "nothing_to_do") return "Rien à ingérer.";
      if (json?.status === "already_running") return "Ingestion déjà en cours.";
      return "État d'ingestion inconnu.";
    case "status": {
      const state = json?.running ? "Ingestion en cours" : "Ingestion à l'arrêt";
      const lr = json?.last_run;
      const lrTxt = json?.running || !lr
        ? ""
        : ` (dernier run : ${lr.ingested ?? "?"} ingéré(s), ${lr.errors ?? "?"} erreur(s))`;
      const blocked = Array.isArray(json?.blocked_files) ? json.blocked_files.length : 0;
      return `${state}${lrTxt}. En attente : ${json?.pending ?? 0}. Bloqués : ${blocked}.`;
    }
    case "tags": {
      const tags = Array.isArray(json?.tags) ? json.tags : [];
      return tags.length ? `Tags : ${tags.join(", ")}.` : "Aucun tag.";
    }
    case "kb_update":
      return "Base de connaissances mise à jour.";
    default:
      return "Réponse wiki vide ou inattendue.";
  }
}
