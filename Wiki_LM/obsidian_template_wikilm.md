<%*
// Template Templater — Interroger Wiki_LM
// Placer dans le dossier Templates configuré dans Templater > Template folder location
// Appeler via : Templater > Create new note from template  OU  raccourci clavier

const WIKI_SERVER = "http://127.0.0.1:5051";

const question = await tp.system.prompt("Question pour Wiki_LM");
if (!question) { return; }

let data;
try {
    const resp = await fetch(`${WIKI_SERVER}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: question, top_k: 5 })
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    data = await resp.json();
} catch (e) {
    new Notice(`Wiki_LM : erreur — ${e.message}`, 8000);
    return;
}

const refs = (data.references || []).map(r => `[[${r}]]`).join(", ");
const block = [
    `## Q : ${question}`,
    ``,
    data.text,
    ``,
    `*Sources : ${refs || "(aucune)"}*`,
].join("\n");

tR += block;
%>
