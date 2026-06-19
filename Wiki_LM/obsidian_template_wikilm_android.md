<%*
// Template Templater — Interroger Wiki_LM depuis Obsidian desktop/Android (via Tailscale)
// Nécessite : server.py lancé sur sanroque
// requestUrl contourne le CSP d'Electron, contrairement à fetch()

const WIKI_SERVER = "http://sanroque:5051";

const mode = await tp.system.suggester(
    ["Hybride (BM25 + sémantique)", "Sémantique", "BM25"],
    ["hybrid", "semantic", "bm25"],
    false,
    "Mode de recherche"
) || "hybrid";

const question = await tp.system.prompt("Question pour Wiki_LM");
if (!question) { return; }

// requestUrl (API Obsidian) contourne le CSP d'Electron, contrairement à fetch().
const { requestUrl } = tp.obsidian ?? require("obsidian");

let data;
try {
    const resp = await requestUrl({
        url: `${WIKI_SERVER}/query`,
        method: "POST",
        contentType: "application/json",
        body: JSON.stringify({ question: question, top_k: 5, mode: mode }),
        throw: false,
    });
    if (resp.status !== 200) throw new Error(`HTTP ${resp.status}`);
    data = resp.json;
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
