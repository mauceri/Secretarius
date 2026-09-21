export interface NoteInput {
  body: string;
  title: string;
  path: string;
}

// La note entière est toujours capturée intégralement, y compris une
// éventuelle section « # Résumé » en tête : c'est ingest.py qui décide,
// côté serveur, d'utiliser cette section pour l'extraction titre/concepts
// par le LLM — le corps de la page src- reste toujours le texte intégral
// (revue du 21/09/2026).
export function buildCaptureText({ body, title, path }: NoteInput): string {
  return `Note d'origine : ${title} (${path})\n\n${body.trim()}`;
}
