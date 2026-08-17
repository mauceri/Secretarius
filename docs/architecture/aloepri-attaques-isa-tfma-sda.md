# Comprendre ISA, TFMA et SDA — les attaques contre AloePri

> Document d'accompagnement de
> `docs/superpowers/specs/2026-08-17-aloepri-poc-sans-attention-design.md`.
> Écrit pour comprendre *pourquoi* le POC est scopé comme il l'est, sans
> prérequis en cryptographie. Source : arXiv 2603.01499v2.

## Le problème qu'AloePri essaie de résoudre

Vous voulez faire tourner un LLM sur un serveur cloud que vous ne contrôlez pas
(RunPod, Modal, un datacenter tiers). Le souci : pour que le serveur calcule
quoi que ce soit, il doit voir vos tokens en entrée et produire des tokens en
sortie. Si le serveur (ou son opérateur, ou quelqu'un qui compromet le serveur)
est malveillant ou simplement curieux, il lit tout en clair.

AloePri propose un tour de passe-passe : le client remplace chaque token par
un autre token, selon une règle secrète connue de lui seul (une **permutation**
du vocabulaire — imaginez un dictionnaire qui dit « le mot n°42 devient le mot
n°8891 »). Le modèle sur le serveur est lui-même modifié pour comprendre
nativement ce vocabulaire permuté — comme s'il avait toujours parlé cette
langue codée. Le serveur calcule normalement, mais ne voit jamais les vrais
tokens : il voit un flux qui n'a de sens que pour qui connaît la permutation.

La question que pose tout schéma de ce genre : **est-ce que le serveur (ou un
attaquant qui l'observe) peut retrouver la permutation, ou au moins deviner
suffisamment de choses sur le texte d'origine sans elle ?** C'est exactement
ce que testent ISA, TFMA et SDA — trois attaques différentes, avec des moyens
différents.

## Les deux familles d'attaquants

Avant de détailler les attaques, la distinction la plus importante : **qui
regarde quoi**.

- **L'attaquant « trafic seul »** — il observe les tokens qui entrent et
  sortent du serveur, sur beaucoup de requêtes, dans le temps. Il n'a pas
  accès aux poids du modèle ni à ce qui se passe *pendant* le calcul. C'est le
  modèle de menace de **TFMA** et **SDA**.

- **L'attaquant « opérateur du serveur »** — c'est littéralement lui qui fait
  tourner le modèle. Il a accès aux poids obfusqués (qu'il possède
  intégralement) et, surtout, à tout ce qui se passe *à l'intérieur* du calcul
  pendant une inférence : les états intermédiaires, les scores d'attention.
  C'est le modèle de menace d'**ISA** (et de deux autres attaques du papier,
  VMA et IA, qui ciblent directement les poids volés plutôt que les états
  internes).

Cette distinction explique tout le reste : une défense qui marche contre l'un
ne marche pas forcément contre l'autre.

## TFMA et SDA — l'attaque du cryptanalyste patient

**TFMA** (Token Frequency-based Matching Attack) part d'une idée vieille comme
la cryptographie : dans n'importe quelle langue, tous les mots ne sont pas
utilisés aussi souvent. « le », « de », « la » reviennent sans arrêt ; un mot
comme « covariant » est rarissime. Si un attaquant observe *beaucoup* de trafic
permuté et compte combien de fois chaque token-permuté apparaît, il peut
comparer ce classement de fréquence à celui d'un texte normal dans la même
langue. Le token-permuté le plus fréquent est probablement « le », le
deuxième « de », etc. C'est exactement comme ça qu'on casse à la main un
« chiffrement par substitution » (chaque lettre remplacée par une autre) dans
les jeux de cryptogrammes — sauf qu'ici l'alphabet n'a pas 26 lettres mais
~150 000 tokens.

**SDA** (Semantic/Sequence-based... — le papier reconstruit carrément des
bouts de texte, pas juste des tokens isolés) va plus loin : une fois une
partie du dictionnaire de fréquence recouvrée, elle essaie de reconstituer des
phrases lisibles.

**Résultat mesuré par le papier**, avec le meilleur attaquant testé (celui qui
connaît le mieux le domaine du texte visé) : 16,5 % des 100 tokens les plus
fréquents retrouvés, et un score de qualité de texte reconstruit (BLEU-4) de
2,1 sur 100 — le papier conclut lui-même que c'est **insuffisant pour former
un texte cohérent**. Autrement dit : même avec beaucoup de trafic et de la
connaissance a priori, cette attaque reste faible.

**Ce qui la rend faible dans son principe** : elle a besoin d'accumuler
beaucoup d'observations pour que les fréquences observées convergent vers les
vraies fréquences. Avec 150 000 tokens possibles et un volume de requêtes
limité (l'usage réel d'un professionnel, pas un serveur qui tourne 24/7 pour
des millions d'utilisateurs), le signal statistique reste bruité longtemps.

**Conséquence pratique** : si on change la permutation périodiquement (une
« rotation »), on limite la quantité de trafic accumulée sous une même clé, ce
qui affaiblit encore cette attaque déjà faible. C'est une défense qui a du
sens contre TFMA/SDA — à condition que le volume de requêtes entre deux
rotations reste raisonnable, et que le coût de re-permuter + retransférer les
poids du modèle reste supportable à cette fréquence.

## ISA — l'attaque de l'opérateur qui regarde par-dessus l'épaule du calcul

**ISA** (Internal State Attack) ne s'intéresse pas au trafic accumulé dans le
temps. Elle exploite le fait que l'opérateur du serveur voit *tout* ce qui se
passe pendant **une seule** inférence — en particulier les **scores
d'attention**, qui mesurent à quel point chaque token du texte « regarde »
chaque autre token. Ces scores dépendent directement des relations entre les
vrais tokens du texte d'origine, même si les tokens eux-mêmes ont été
permutés : la *structure* des relations, elle, ne ment pas.

Le papier montre (Tableau 4 de l'article) l'effet cumulatif de chaque
protection contre cette attaque, mesuré en « taux de récupération » (TTRSR,
plus c'est bas mieux c'est) :

| Protection appliquée | Taux de récupération |
|---|---|
| Bruit sur l'embedding seul | 87,1 % |
| + matrices clés sur l'embedding | 87,1 % (inchangé) |
| + permutations sur l'attention (têtes et blocs) | **0,0 %** |

Le chiffre qui compte : **sans toucher à la couche d'attention, 87 % de fuite,
quelles que soient les protections mises sur l'embedding.** La couche
d'attention est celle qui referme ce trou spécifique — les protections sur
l'embedding (bruit, matrices clés) ne le touchent pas du tout, parce que le
problème n'est pas dans l'embedding, il est dans la façon dont l'attention
laisse transparaître les relations entre tokens.

**Correction (2026-08-17)** : une version précédente de ce document affirmait
que ISA fonctionnait « dès une seule requête, sans rien d'autre » et qu'aucune
rotation ne pouvait donc s'en protéger. C'était trop catégorique — objection
justifiée en relecture. Le papier qualifie ISA de **« training-based
inversion »** (même famille que IMA, Inversion Model Attack) : ce n'est pas
une lecture directe d'une requête inconnue, mais un **modèle d'inversion
entraîné au préalable** par l'attaquant, probablement en envoyant lui-même des
requêtes-sondes de contenu connu au service qu'il opère, pour calibrer la
correspondance (état interne observé) → (texte). Une fois ce modèle entraîné,
il peut être appliqué à de nouvelles requêtes, potentiellement rapidement.

**Conséquence corrigée** : une rotation de la permutation/des clés invaliderait
ce modèle d'inversion entraîné sur l'ancienne configuration — la rotation
**pourrait** donc offrir une protection contre ISA, contrairement à ce
qu'affirmait la version précédente. Ce qui reste incertain (non résolu par ce
que nous avons pu extraire du papier — méthode détaillée et référence [10]
non accessibles) : le volume de requêtes-sondes nécessaire pour entraîner ce
modèle d'inversion, et donc la fréquence de rotation qu'il faudrait pour
rester devant l'attaquant. C'est la même question ouverte que pour TFMA/SDA
(« quel volume avant que l'attaque devienne efficace »), appliquée à un signal
différent (états internes entraînés plutôt que fréquence de tokens).

## Pourquoi cette distinction compte pour Secretarius

Deux cas d'usage bien différents :

1. **Protéger un gros volume de documents/conversations dans le temps** contre
   un observateur qui accumule du trafic → TFMA/SDA est la menace pertinente,
   et rotation + permutation embedding (sans attention) pourrait suffire.

2. **Protéger le contenu d'une question précise, posée une fois**, contre
   l'opérateur du serveur qui la traite → ISA est la menace pertinente.
   L'obfuscation de l'attention est la seule protection **mesurée** par le
   papier (0 % au Tableau 4). La rotation est une protection **candidate**
   mais non quantifiée : elle invaliderait le modèle d'inversion entraîné par
   l'attaquant, à condition de tourner plus vite qu'il ne peut le
   recalibrer — coût de calibration inconnu à ce stade (cf. correction
   ci-dessus).

C'est le deuxième cas qui correspond à la remarque de l'avocat d'affaires :
les documents sont de toute façon déjà dans le cloud (peu de valeur ajoutée à
les protéger davantage), mais **la question posée à l'IA à leur sujet** est
l'information réellement sensible.

## Ce que ça implique pour le POC en cours

Le POC scopé (`2026-08-17-aloepri-poc-sans-attention-design.md`) se limite
volontairement à l'embedding/unembedding et au FFN, sans attention — sans
rotation non plus. Il ne protège donc, tel quel, ni contre TFMA/SDA (pas de
rotation) ni avec certitude contre ISA (pas d'attention, et rotation non
implémentée). Ce n'est pas un problème pour ce que le POC mesure (mécanique,
qualité, vitesse), mais c'est la raison pour laquelle il n'est présenté nulle
part comme une solution de confidentialité utilisable en l'état.

Si le POC est concluant sur qualité et vitesse, deux chemins restent ouverts
pour la suite, pas un seul : **l'obfuscation d'attention** (protection
mesurée par le papier, coût de développement plus élevé) et **la rotation de
clé/permutation** (protection non quantifiée mais potentiellement moins
coûteuse à construire, à condition de caractériser d'abord le coût de
calibration de l'attaquant contre ISA/IMA — extension possible de l'Étape 0).
Le choix entre les deux (ou leur combinaison) reste à trancher une fois ces
inconnues réduites, pas figé par ce document.

## VMA et IA — mentionnées pour être complet

Le papier teste deux autres attaques contre l'opérateur du serveur : **VMA**
(Vocabulary-Matching Attack) et **IA** (Invariant Attack). Toutes deux
cherchent à retrouver la permutation en comparant la structure des poids
obfusqués volés à un modèle public non obfusqué du même type (par exemple, si
un attaquant a une copie de Qwen2.5-7B non modifié, il peut essayer de
recaler les lignes de la matrice d'embedding obfusquée sur celles du modèle
public en cherchant quelles paires de lignes ont des relations similaires).
Ces attaques ciblent les poids directement, pas le trafic ni les états
internes — elles ne sont pas au cœur de ce POC mais font partie du tableau
d'ensemble si vous creusez le papier plus tard.
