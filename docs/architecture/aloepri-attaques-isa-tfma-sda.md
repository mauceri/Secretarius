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

**Ce qu'on ne sait pas avec certitude** (honnêteté sur les limites de cette
lecture) : le papier ne précise pas si ce taux de 87 % nécessite d'observer
plusieurs requêtes ou s'il s'obtient dès une seule. Le nom de l'attaque
(« état interne », pas « fréquence dans le temps ») et la nature du signal
exploité (une structure présente dans *chaque* inférence individuelle, pas une
statistique qui s'accumule) suggèrent fortement que **ça marche dès une seule
requête** — mais ce n'est pas explicitement confirmé dans ce que nous avons pu
extraire du papier.

**Conséquence pratique** : si c'est bien le cas, **aucune fréquence de
rotation ne protège contre ISA**. Rotation ou pas, chaque requête individuelle
expose sa propre structure interne à l'opérateur, dans l'instant où elle est
traitée. La rotation réduit le trafic *accumulé* sous une clé — elle ne réduit
en rien ce qu'une seule requête révèle en elle-même.

## Pourquoi cette distinction compte pour Secretarius

Deux cas d'usage bien différents :

1. **Protéger un gros volume de documents/conversations dans le temps** contre
   un observateur qui accumule du trafic → TFMA/SDA est la menace pertinente,
   et rotation + permutation embedding (sans attention) pourrait suffire.

2. **Protéger le contenu d'une question précise, posée une fois**, contre
   l'opérateur du serveur qui la traite → ISA est la menace pertinente, et
   *seule* l'obfuscation de l'attention semble en protéger, d'après le
   Tableau 4.

C'est le deuxième cas qui correspond à la remarque de l'avocat d'affaires :
les documents sont de toute façon déjà dans le cloud (peu de valeur ajoutée à
les protéger davantage), mais **la question posée à l'IA à leur sujet** est
l'information réellement sensible — et c'est une requête individuelle, pas un
flux qu'on peut diluer par rotation.

## Ce que ça implique pour le POC en cours

Le POC scopé (`2026-08-17-aloepri-poc-sans-attention-design.md`) se limite
volontairement à l'embedding/unembedding et au FFN, sans attention — il ne
protège donc *a priori* pas contre ISA, seulement (potentiellement) contre
TFMA/SDA. Ce n'est pas un problème pour ce qu'il mesure (mécanique, qualité,
vitesse), mais c'est la raison pour laquelle il n'est présenté nulle part
comme une solution de confidentialité utilisable en l'état. Si le POC est
concluant sur qualité et vitesse, l'étape suivante (l'obfuscation d'attention)
n'est pas un raffinement optionnel : c'est la pièce qui manque pour répondre
au vrai besoin (confidentialité d'une requête isolée), pas une amélioration
marginale d'un schéma déjà utilisable.

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
