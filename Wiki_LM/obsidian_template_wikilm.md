<%*
// Template Templater — Interroger Wiki_LM
// Placer dans le dossier Templates configuré dans Templater > Template folder location
// Appeler via : Templater > Open Insert Template modal (PAS "Create new note
// from template" — ce template n'insère plus rien dans la note en cours ;
// utiliser "Create new note" laisserait une note vide derrière lui).
//
// N'insère plus la réponse dans la note en cours : le serveur écrit lui-même
// un enregistrement horodaté dans Wiki_LM/historique/, ce template se
// contente de l'ouvrir dans un nouvel onglet.

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
