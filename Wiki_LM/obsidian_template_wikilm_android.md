<%*
// Template Templater — Interroger Wiki_LM depuis Obsidian desktop/Android (via Tailscale)
// Nécessite : server.py lancé sur sanroque
// requestUrl contourne le CSP d'Electron, contrairement à fetch()
//
// N'insère plus la réponse dans la note en cours : le serveur écrit lui-même
// un enregistrement horodaté dans Wiki_LM/historique/, ce template se
// contente de l'ouvrir dans un nouvel onglet. Appeler via Templater > Open
// Insert Template modal (pas "Create new note from template").

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

const historyPath = `Wiki_LM/historique/${data.history_slug}.md`;
let file = app.vault.getAbstractFileByPath(historyPath);
for (let i = 0; i < 10 && !file; i++) {
    // Le serveur vient d'écrire ce fichier hors du cache Obsidian ; laisser
    // le temps au watcher (et à la synchro sur mobile) de le voir apparaître.
    await new Promise((r) => setTimeout(r, 200));
    file = app.vault.getAbstractFileByPath(historyPath);
}
if (file) {
    await app.workspace.getLeaf(true).openFile(file);
} else {
    new Notice(`Wiki_LM : note d'historique introuvable (${historyPath})`, 8000);
}
%>
