# Wiki LLM — Un patron pour bases de connaissances personnelles

Un patron pour construire des bases de connaissances personnelles à l'aide de modèles de langage.

Ce document est conçu pour être partagé avec votre agent LLM (Claude Code, OpenClaw, ou autre).
Son but est de communiquer l'idée générale ; votre agent construira les détails en collaboration avec vous.

---

## L'idée centrale

La plupart des gens découvrent les LLM à travers le RAG : on dépose une collection de fichiers, le modèle récupère les passages pertinents au moment de la question et génère une réponse. Ça fonctionne, mais le modèle redécouvre la connaissance à partir de zéro à chaque question. Rien ne s'accumule. Posez une question subtile qui nécessite de synthétiser cinq documents, et le modèle doit retrouver et assembler les fragments pertinents à chaque fois. NotebookLM, les fichiers uploadés dans ChatGPT, et la plupart des systèmes RAG fonctionnent ainsi.

L'idée ici est différente. Plutôt que de simplement récupérer des passages dans des documents bruts au moment de la requête, le LLM **construit et maintient de façon incrémentale un wiki persistant** — une collection structurée et interconnectée de fichiers Markdown qui s'interpose entre vous et les sources brutes. Quand vous ajoutez une nouvelle source, le modèle ne se contente pas de l'indexer pour plus tard. Il la lit, en extrait les informations clés, et les intègre dans le wiki existant — en mettant à jour les pages d'entités, en révisant les synthèses thématiques, en notant là où les nouvelles données contredisent les anciennes, en renforçant ou challengeant la synthèse en cours. La connaissance est compilée une fois puis *maintenue à jour*, et non redérivée à chaque requête.

C'est la différence fondamentale : **le wiki est un artefact persistant et cumulatif.** Les références croisées sont déjà en place. Les contradictions ont déjà été signalées. La synthèse reflète déjà tout ce que vous avez lu. Le wiki s'enrichit à chaque source ajoutée et à chaque question posée.

Vous n'écrivez jamais (ou presque jamais) le wiki vous-même — le LLM l'écrit et le maintient entièrement. Vous êtes responsable des sources, de l'exploration et des bonnes questions. Le LLM se charge de tout le travail de fond — résumer, établir des références croisées, classer, tenir à jour — ce qui rend une base de connaissances réellement utile dans le temps. En pratique : l'agent LLM est ouvert d'un côté, Obsidian de l'autre. Le LLM modifie le wiki au fil de la conversation, et vous parcourez les résultats en temps réel — en suivant les liens, en consultant la vue graphe, en lisant les pages mises à jour. Obsidian est l'environnement de développement ; le LLM est le programmeur ; le wiki est le code source.

Ce patron s'applique à de nombreux contextes :

- **Personnel** : suivre ses propres objectifs, sa santé, sa psychologie, son développement — archiver des entrées de journal, des articles, des notes de podcasts, et construire progressivement une image structurée de soi-même.
- **Recherche** : approfondir un sujet sur des semaines ou des mois — lire des articles, des papiers, des rapports, et construire incrémentalement un wiki exhaustif avec une thèse en évolution.
- **Lecture d'un livre** : archiver chaque chapitre au fur et à mesure, développer des pages pour les personnages, les thèmes, les fils narratifs, et leurs interconnexions. À la fin vous disposez d'un wiki compagnon riche. Pensez aux wikis de fans comme Tolkien Gateway — des milliers de pages interconnectées couvrant personnages, lieux, événements et langues, construites par une communauté de bénévoles au fil des années. Vous pourriez construire quelque chose de similaire personnellement en lisant, avec le LLM qui gère toutes les références croisées et la maintenance.
- **Entreprise / équipe** : un wiki interne maintenu par des LLM, alimenté par des fils Slack, des transcriptions de réunions, des documents de projet, des appels clients. Éventuellement avec des humains qui valident les mises à jour. Le wiki reste à jour parce que le LLM fait la maintenance que personne dans l'équipe ne veut faire.
- **Veille concurrentielle, due diligence, planification de voyage, notes de cours, passions** — tout contexte où vous accumulez de la connaissance dans le temps et voulez qu'elle soit organisée plutôt qu'éparpillée.

---

## Architecture

Trois couches :

**Sources brutes** — votre collection de documents source. Articles, papiers, images, fichiers de données. Ces sources sont immuables — le LLM les lit mais ne les modifie jamais. C'est votre référence absolue.

**Le wiki** — un répertoire de fichiers Markdown générés par le LLM. Résumés, pages d'entités, pages de concepts, comparaisons, vue d'ensemble, synthèse. Le LLM possède entièrement cette couche. Il crée les pages, les met à jour à l'arrivée de nouvelles sources, maintient les références croisées, et assure la cohérence globale. Vous lisez ; le LLM écrit.

**Le schéma** — un document (par exemple `CLAUDE.md` pour Claude Code, ou `AGENTS.md` pour Codex) qui indique au LLM comment le wiki est structuré, quelles sont les conventions, et quels flux de travail suivre pour ingérer des sources, répondre aux questions ou maintenir le wiki. C'est le fichier de configuration central — c'est lui qui fait du LLM un mainteneur de wiki discipliné plutôt qu'un chatbot générique. Vous et le LLM faites évoluer ce document ensemble au fil du temps, en affinant ce qui fonctionne pour votre domaine.

---

## Opérations

**Ingestion.** Vous déposez une nouvelle source dans la collection brute et demandez au LLM de la traiter. Un exemple de flux : le LLM lit la source, discute des points clés avec vous, écrit une page de résumé dans le wiki, met à jour l'index, met à jour les pages d'entités et de concepts concernées dans tout le wiki, et ajoute une entrée au journal. Une seule source peut toucher 10 à 15 pages du wiki. Personnellement, je préfère ingérer les sources une par une en restant impliqué — je lis les résumés, vérifie les mises à jour, et guide le LLM sur ce qu'il faut mettre en avant. Mais vous pouvez aussi ingérer en lot de nombreuses sources à la fois avec moins de supervision. C'est à vous de développer le flux de travail qui correspond à votre style et de le documenter dans le schéma pour les prochaines sessions.

**Interrogation.** Vous posez des questions au wiki. Le LLM recherche les pages pertinentes, les lit, et synthétise une réponse avec des citations. Les réponses peuvent prendre différentes formes selon la question — une page Markdown, un tableau comparatif, un diaporama (Marp), un graphique (matplotlib). L'insight important : **les bonnes réponses peuvent être archivées dans le wiki comme nouvelles pages.** Une comparaison que vous avez demandée, une analyse, une connexion que vous avez découverte — tout cela a de la valeur et ne doit pas disparaître dans l'historique de chat. Vos explorations s'accumulent ainsi dans la base de connaissances, tout comme les sources ingérées.

**Audit.** Périodiquement, demandez au LLM de vérifier l'état de santé du wiki. Points à examiner : contradictions entre pages, affirmations périmées contredites par des sources récentes, pages orphelines sans liens entrants, concepts importants mentionnés mais sans page propre, références croisées manquantes, lacunes comblables par une recherche web. Le LLM est efficace pour suggérer de nouvelles questions à explorer et de nouvelles sources à chercher. Cela maintient le wiki en bonne santé à mesure qu'il grossit.

---

## Indexation et journal

Deux fichiers spéciaux aident le LLM (et vous) à naviguer dans le wiki à mesure qu'il grandit. Ils ont des rôles distincts :

**index.md** est orienté contenu. C'est un catalogue de tout ce qui est dans le wiki — chaque page listée avec un lien, un résumé d'une ligne, et éventuellement des métadonnées comme la date ou le nombre de sources. Organisé par catégorie (entités, concepts, sources, etc.). Le LLM le met à jour à chaque ingestion. Quand il répond à une requête, il lit d'abord l'index pour trouver les pages pertinentes, puis les explore. Cela fonctionne remarquablement bien à échelle modérée (~100 sources, ~quelques centaines de pages) et évite d'avoir besoin d'une infrastructure RAG basée sur des embeddings.

**log.md** est chronologique. C'est un registre en ajout seul de ce qui s'est passé et quand — ingestions, requêtes, audits. Une astuce utile : si chaque entrée commence par un préfixe cohérent (par exemple `## [2026-04-02] ingest | Titre de l'article`), le journal devient analysable avec de simples outils Unix — `grep "^## \[" log.md | tail -5` donne les 5 dernières entrées. Le journal fournit une chronologie de l'évolution du wiki et aide le LLM à comprendre ce qui a été fait récemment.

---

## En option : outils en ligne de commande

À un certain stade, vous voudrez peut-être construire de petits outils pour aider le LLM à opérer sur le wiki plus efficacement. Un moteur de recherche sur les pages du wiki est le plus évident — à petite échelle le fichier index suffit, mais quand le wiki grandit vous voulez une vraie recherche. [qmd](https://github.com/tobi/qmd) est une bonne option : c'est un moteur de recherche local pour fichiers Markdown avec recherche hybride BM25/vectorielle et reclassement par LLM, entièrement sur machine. Il dispose d'une interface en ligne de commande (pour que le LLM puisse l'appeler en shell) et d'un serveur MCP (pour que le LLM l'utilise comme outil natif). Vous pouvez aussi construire quelque chose de plus simple — le LLM peut vous aider à prototyper un script de recherche naïf selon les besoins.

---

## Conseils pratiques

- **Obsidian Web Clipper** est une extension de navigateur qui convertit les articles web en Markdown. Très utile pour intégrer rapidement des sources dans votre collection brute.
- **Téléchargez les images localement.** Dans Paramètres Obsidian → Fichiers et liens, définissez le chemin du dossier des pièces jointes vers un répertoire fixe (par exemple `raw/assets/`). Puis dans Paramètres → Raccourcis clavier, cherchez « Télécharger » pour trouver « Télécharger les pièces jointes du fichier courant » et associez un raccourci (par exemple Ctrl+Maj+D). Après avoir clipé un article, utilisez le raccourci et toutes les images sont téléchargées sur le disque local. C'est optionnel mais utile — cela permet au LLM de voir et référencer les images directement plutôt que de dépendre d'URL qui peuvent se casser.
- **La vue graphe d'Obsidian** est la meilleure façon de visualiser la forme de votre wiki — ce qui est connecté à quoi, quelles pages sont des hubs, lesquelles sont orphelines.
- **Marp** est un format de diaporama basé sur Markdown. Obsidian dispose d'un plugin pour ça. Utile pour générer des présentations directement depuis le contenu du wiki.
- **Dataview** est un plugin Obsidian qui exécute des requêtes sur les métadonnées frontmatter des pages. Si votre LLM ajoute du YAML frontmatter aux pages du wiki (étiquettes, dates, nombre de sources), Dataview peut générer des tableaux et des listes dynamiques.
- Le wiki n'est qu'un dépôt git de fichiers Markdown. Vous obtenez l'historique des versions, les branches et la collaboration gratuitement.

---

## Pourquoi ça fonctionne

La partie fastidieuse de la maintenance d'une base de connaissances n'est ni la lecture ni la réflexion — c'est la tenue à jour. Mettre à jour les références croisées, garder les résumés à jour, noter quand de nouvelles données contredisent des affirmations anciennes, maintenir la cohérence sur des dizaines de pages. Les humains abandonnent les wikis parce que la charge de maintenance croît plus vite que la valeur produite. Les LLM ne s'ennuient pas, n'oublient pas de mettre à jour une référence croisée, et peuvent toucher 15 fichiers en un seul passage. Le wiki reste maintenu parce que le coût de la maintenance est quasi nul.

Le rôle humain est de sélectionner les sources, diriger l'analyse, poser de bonnes questions, et réfléchir à ce que tout cela signifie. Le rôle du LLM est tout le reste.

L'idée est apparentée en esprit au Memex de Vannevar Bush (1945) — un réservoir de connaissance personnel et sélectionné, avec des pistes associatives entre documents. La vision de Bush était plus proche de ceci que de ce qu'est devenu le web : privée, activement sélectionnée, avec les connexions entre documents aussi précieuses que les documents eux-mêmes. Ce qu'il n'avait pas résolu, c'était : qui fait la maintenance ? Le LLM s'en charge.

---

## Note

Ce document est intentionnellement abstrait. Il décrit l'idée, pas une implémentation spécifique. La structure exacte des répertoires, les conventions du schéma, les formats de pages, l'outillage — tout cela dépendra de votre domaine, de vos préférences et de votre LLM. Tout ce qui est mentionné ci-dessus est optionnel et modulaire — prenez ce qui est utile, ignorez le reste. Par exemple : vos sources peuvent être uniquement textuelles, donc vous n'avez pas besoin de gestion des images. Votre wiki peut être suffisamment petit pour que le fichier index soit tout ce dont vous avez besoin, sans moteur de recherche. Vous ne vous intéressez peut-être pas aux diaporamas et voulez simplement des pages Markdown. Vous voudrez peut-être un ensemble de formats de sortie complètement différent. La bonne façon d'utiliser ce document est de le partager avec votre agent LLM et de travailler ensemble pour instancier une version adaptée à vos besoins. Ce document a pour seul rôle de communiquer le patron. Votre LLM peut déduire le reste.
