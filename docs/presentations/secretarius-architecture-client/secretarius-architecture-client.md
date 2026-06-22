---
marp: true
theme: default
paginate: true
size: 16:9
---

<style>
section {
  font-size: 22px;
}
table {
  font-size: 18px;
}
blockquote {
  font-size: 22px;
}
pre {
  font-size: 16px;
}
header {
  font-size: 14px;
  color: #999;
}
header strong {
  color: #2563eb;
}
section.title-slide {
  background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
}
section.part-constat,
section.part-solution,
section.part-engagements,
section.title-slide {
  --h1-color: #fff;
  --heading-strong-color: #fff;
  --fgColor-default: rgba(255, 255, 255, 0.95);
  --fgColor-muted: rgba(255, 255, 255, 0.7);
  color: white;
}
section.part-constat {
  background: linear-gradient(135deg, #1e3a5f 0%, #2d5a8e 100%);
}
section.part-solution {
  background: linear-gradient(135deg, #064e3b 0%, #047857 100%);
}
section.part-engagements {
  background: linear-gradient(135deg, #7a4a1a 0%, #a66b2e 100%);
}
.two-cols {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 38px;
  align-items: center;
}
.two-cols-wide {
  display: grid;
  grid-template-columns: 0.85fr 1.15fr;
  gap: 34px;
  align-items: center;
}
.illustration {
  width: 100%;
  max-height: 460px;
  object-fit: contain;
}
.big-number {
  font-size: 72px;
  line-height: 1;
  font-weight: 700;
  color: #1e3a5f;
}
.metric {
  padding: 20px;
  border-radius: 16px;
  background: #f4f7fa;
  text-align: center;
}
.metric strong {
  display: block;
  font-size: 42px;
  color: #047857;
}
.flow {
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 14px;
  margin-top: 34px;
}
.flow-box {
  min-width: 180px;
  padding: 18px 14px;
  border: 2px solid #2d5a8e;
  border-radius: 14px;
  background: #f8fafc;
  text-align: center;
}
.arrow {
  font-size: 34px;
  color: #64748b;
}
.timeline {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 20px;
  margin-top: 30px;
}
.timeline > div {
  padding: 22px;
  border-top: 6px solid #047857;
  background: #f4f7fa;
}
.small {
  font-size: 17px;
  color: #64748b;
}
</style>

<!-- _class: lead title-slide -->

# Secretarius change d’architecture
## Pourquoi cette refonte était nécessaire

**Objectif**: un assistant plus fiable, confidentiel et durable
**Contexte**: point d’avancement et explication du retard
**Date**: Juin 2026

---

# Les questions auxquelles nous allons répondre

1. **Pourquoi l’architecture initiale a-t-elle atteint ses limites ?**
2. **Pourquoi cette limite affecte-t-elle particulièrement les modèles légers ?**
3. **Quelle architecture avons-nous retenue ?**
4. **Pourquoi cette refonte a-t-elle un impact sur le calendrier ?**
5. **Quels bénéfices concrets apporte-t-elle au produit livré ?**

---

<!-- _class: lead title-slide -->

# Agenda

### Partie 1: Le constat
Comprendre la limite rencontrée et ses conséquences

### Partie 2: La solution retenue
Présenter une architecture légère, spécialisée et contrôlable

### Partie 3: Les engagements
Expliquer le retard, les bénéfices et la suite du projet

---

<!-- _header: "" -->
<!-- _class: lead part-constat -->

# Partie 1: Le constat

**L’architecture initiale fonctionnait, mais ne pouvait pas évoluer durablement**

---

<!-- header: "**Le constat** > La solution retenue > Les engagements" -->

# Ce qui était prévu

Secretarius devait réunir dans un même assistant :

- Une conversation simple en langage naturel
- Une mémoire documentaire personnelle
- La recherche dans les courriels et les documents
- La consultation sécurisée de sources externes
- L’exécution de tâches selon les règles de l’utilisateur

> L’objectif reste inchangé : un assistant personnel utile, sobre et sous contrôle

---

# Le point de blocage

<div class="two-cols-wide">
<div>

L’assistant principal devait connaître en permanence la description de **tous les outils disponibles**

- Recherche documentaire
- Messagerie et agenda
- Navigation sécurisée
- Base de connaissances
- Outils de contrôle et de routage

Chaque nouvel outil augmentait la quantité d’instructions à relire avant même de répondre

</div>
<div>

![Architecture initiale surchargée](assets/architecture-surchargee.png)

</div>
</div>

---

# Un contexte devenu trop lourd

<div class="two-cols">
<div>

<div class="big-number">≈ 11 700</div>

**unités de contexte** chargées au démarrage de chaque échange

Cette charge est présente avant même d’ajouter :

- La demande de l’utilisateur
- Les documents utiles
- L’historique de la conversation

</div>
<div>

| Conséquence | Effet observé |
|---|---|
| Démarrage plus lent | Temps d’attente accru |
| Capacité utile réduite | Moins de place pour le dossier réel |
| Routage fragile | Mauvais outil ou mauvaise opération |
| Modèle léger saturé | Réponses instables ou impossibles |

</div>
</div>

---

# Pourquoi un modèle léger ne suffit plus

Un modèle léger peut être rapide, économique et exécuté localement

Mais il ne peut pas simultanément :

- Relire un volumineux catalogue d’outils
- Comprendre la demande métier
- Choisir le bon outil et les bons paramètres
- Contrôler le résultat et répondre clairement

> Le problème ne venait pas d’un manque de puissance brute, mais d’une mauvaise répartition du travail

---

# Continuer ainsi aurait créé une dette durable

| Option | Avantage immédiat | Risque à terme |
|---|---|---|
| Ajouter un modèle plus puissant | Masque le problème | Coût, dépendance et confidentialité |
| Réduire quelques instructions | Gain rapide mais limité | Saturation au prochain outil |
| Repenser l’architecture | Effort de refonte | Base saine et extensible |

**Décision retenue**: corriger la structure plutôt que compenser ses limites

---

<!-- _header: "" -->
<!-- _class: lead part-solution -->

# Partie 2: La solution retenue

**Un chef d’orchestre léger entouré de spécialistes**

---

<!-- header: "Le constat > **La solution retenue** > Les engagements" -->

# Le principe directeur

<div class="two-cols">
<div>

L’assistant principal devient un **orchestrateur léger**

Son rôle se limite à :

- Comprendre la demande
- Identifier l’action attendue
- Confier le travail au bon spécialiste
- Restituer le résultat à l’utilisateur

</div>
<div>

<div class="metric">
Avant
<strong>≈ 11 700</strong>
unités de contexte
</div>

<div style="height: 18px"></div>

<div class="metric">
Cible
<strong>≈ 1 000</strong>
unités de contexte
</div>

</div>
</div>

---

# Une équipe d’agents spécialisés

![Architecture modulaire avec orchestrateur léger](assets/architecture-modulaire.png)

Chaque agent dispose uniquement des capacités nécessaires à sa mission

---

# Une responsabilité claire par agent

| Composant | Mission | Accès |
|---|---|---|
| **Tiron** | Dialogue et orchestration | Minimum nécessaire |
| **Agent wiki** | Mémoire et recherche documentaire | Base de connaissances |
| **Scout** | Lecture sécurisée du web | Sources externes filtrées |
| **Agent Google** | Courriels, agenda et Drive | Services Google autorisés |

Cette séparation applique un principe simple :

> Un composant ne reçoit que les outils et les données dont il a réellement besoin

---

# Les outils sortent du contexte permanent

Dans l’ancienne architecture, les outils MCP devaient être décrits au modèle principal

Dans la nouvelle architecture :

- Les capacités sont placées dans l’environnement du spécialiste
- Les instructions détaillées sont chargées seulement au moment utile
- L’orchestrateur ne porte plus le catalogue complet
- Chaque environnement peut être testé et mis à jour séparément

**Résultat**: moins de contexte, moins de dépendances et moins de points de panne

---

# Un routage déterministe

Pour les opérations importantes, Secretarius ne demande plus au modèle de deviner librement quoi faire

<div class="flow">
  <div class="flow-box"><strong>Commande ou intention</strong><br><span class="small">Exemple : rechercher un mail</span></div>
  <div class="arrow">→</div>
  <div class="flow-box"><strong>Route définie</strong><br><span class="small">Opération précise</span></div>
  <div class="arrow">→</div>
  <div class="flow-box"><strong>Agent spécialisé</strong><br><span class="small">Outil isolé</span></div>
  <div class="arrow">→</div>
  <div class="flow-box"><strong>Résultat fidèle</strong><br><span class="small">Sans succès inventé</span></div>
</div>

> L’intelligence reste utile pour comprendre et dialoguer, mais les opérations critiques suivent un chemin contrôlé

---

# Ce que la nouvelle architecture améliore

| Dimension | Amélioration |
|---|---|
| **Fiabilité** | Une opération correspond à un chemin connu |
| **Confidentialité** | L’orchestrateur peut fonctionner avec un modèle local |
| **Sécurité** | Chaque agent possède des accès limités |
| **Performance** | Le contexte initial est fortement réduit |
| **Maintenance** | Les spécialistes évoluent indépendamment |
| **Évolutivité** | Une nouvelle capacité n’alourdit plus tout le système |

---

<!-- _header: "" -->
<!-- _class: lead part-engagements -->

# Partie 3: Les engagements

**Le retard finance une base plus fiable, pas une simple modification cosmétique**

---

<!-- header: "Le constat > La solution retenue > **Les engagements**" -->

# Pourquoi cette refonte prend du temps

<div class="two-cols-wide">
<div>

![Refonte des fondations](assets/refonte-fondations.png)

</div>
<div>

La modification touche les fondations techniques :

- Séparation des responsabilités
- Création des agents spécialisés
- Isolation de leurs outils
- Nouveau mécanisme de routage
- Vérification des échanges entre agents
- Reprise des fonctions déjà développées

> Une partie du travail existant doit être réintégrée et retestée dans la nouvelle structure

</div>
</div>

---

# L’impact sur le calendrier

<div class="timeline">
<div>

### 1. Limite identifiée

Les tests avec un modèle léger révèlent que le contexte des outils est incompatible avec l’objectif de sobriété

</div>
<div>

### 2. Architecture sécurisée

Les preuves techniques valident l’orchestrateur léger, la délégation et le routage déterministe

</div>
<div>

### 3. Recouvrement fonctionnel

Les fonctions existantes sont réintégrées, testées et préparées pour la livraison

</div>
</div>

Le retard provient principalement de cette **reconstruction contrôlée**, nécessaire pour éviter une livraison fragile

---

# Ce que le client y gagne

- **Un assistant plus stable** dans les tâches quotidiennes
- **Des réponses plus rapides** grâce à un contexte réduit
- **Une meilleure confidentialité** avec la possibilité d’orchestrer localement
- **Des actions plus sûres** grâce aux routes et validations explicites
- **Une solution durable** pouvant accueillir de nouvelles fonctions métier

Pour un usage immobilier, cela prépare notamment une gestion plus fiable des dossiers, documents, courriels et recherches

---

# Nos engagements pour la suite

1. **Prioriser le recouvrement fonctionnel** avant toute nouvelle fonctionnalité
2. **Tester chaque commande de bout en bout** dans les conditions réelles d’utilisation
3. **Conserver une validation humaine** pour les actions externes sensibles
4. **Documenter clairement l’état d’avancement** et les limites restantes
5. **Livrer sur une base stabilisée** plutôt que masquer une fragilité connue

---

# À retenir

La première architecture concentrait trop de responsabilités dans un seul modèle

La nouvelle architecture :

- Allège l’orchestrateur d’environ **11 700 à 1 000 unités de contexte**
- Confie chaque mission à un agent spécialisé
- Remplace les décisions fragiles par des routes contrôlées
- Renforce la confidentialité, la sécurité et la maintenabilité

> Le calendrier s’allonge à court terme pour éviter des retards et incidents répétés après la livraison

