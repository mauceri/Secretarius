<%*
// Template Templater — Interroger Wiki_LM depuis Obsidian desktop/Android (via Tailscale)
// Nécessite : server.py lancé sur sanroque
// requestUrl contourne le CSP d'Electron, contrairement à fetch()

const WIKI_SERVER = "http://sanroque:5051";

const question = await tp.system.prompt("Question pour Wiki_LM");
if (!question) { return; }

let data;
try {
    const resp = await fetch(`${WIKI_SERVER}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: question, top_k: 5 }),
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
