export interface NoteInput {
  body: string;
  title: string;
  path: string;
}

const INCIPIT_LENGTH = 200;
const SUMMARY_HEADINGS = ["résumé", "summary"];

export function buildCaptureText({ body, title, path }: NoteInput): string {
  const content = extractContent(body);
  return `Note d'origine : ${title} (${path})\n\n${content}`;
}

function extractContent(body: string): string {
  const summary = extractSummarySection(body);
  return summary !== null ? summary : truncateIncipit(body);
}

function extractSummarySection(body: string): string | null {
  const lines = body.split("\n");
  let i = 0;
  while (i < lines.length && lines[i].trim() === "") i++;
  if (i >= lines.length) return null;

  const headingMatch = lines[i].match(/^#\s+(.+)$/);
  if (!headingMatch) return null;
  if (!SUMMARY_HEADINGS.includes(headingMatch[1].trim().toLowerCase())) return null;

  const sectionLines: string[] = [];
  for (let j = i + 1; j < lines.length; j++) {
    if (/^#{1,6}\s+/.test(lines[j])) break;
    sectionLines.push(lines[j]);
  }
  return sectionLines.join("\n").trim();
}

function truncateIncipit(body: string): string {
  const trimmed = body.trim();
  if (trimmed.length <= INCIPIT_LENGTH) return trimmed;
  const slice = trimmed.slice(0, INCIPIT_LENGTH);
  const lastSpace = slice.lastIndexOf(" ");
  const cut = lastSpace > 0 ? slice.slice(0, lastSpace) : slice;
  return `${cut}…`;
}
