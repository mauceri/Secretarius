# Comprendre ISA, TFMA et SDA — les attaques contre AloePri

> Document d'accompagnement de
> `docs/superpowers/specs/2026-08-17-aloepri-poc-sans-attention-design.md`.
> Écrit pour comprendre *pourquoi* le POC est scopé comme il l'est, sans
> prérequis en cryptographie. Source : arXiv 2603.01499v2.
>
> **Note méthodologique** : les premières versions de ce document s'appuyaient
> sur des résumés générés à la demande (outil de fetch web) plutôt que sur une
> lecture directe du PDF. Ça a produit deux erreurs successives, corrigées
> en cours de route (voir la section ISA) — dont une contradiction directe
> entre deux résumés de la même table. La section ISA a depuis été relue sur
> le PDF téléchargé (Tableau 4 page 15, Appendix D.1 page 27) et est fiable ;
> le reste du document n'a pas fait l'objet de la même relecture systématique.

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

**Troisième et dernière correction (2026-08-17)** — après deux corrections
successives fondées sur des extractions résumées contradictoires (une requête
attribuait certains chiffres à « AttnScore », une autre aux mêmes chiffres
mais sous « HiddenState »), le texte primaire a été lu directement (PDF
téléchargé, Tableau 4 lu en image page 15, algorithme ISA lu en texte page 27,
Appendix D.1) — plus de couche de résumé automatique entre le papier et cette
lecture.

**Structure réelle du Tableau 4** : TTRSR est mesuré de deux façons
distinctes contre ISA, pas trois métriques séparées comme les extractions
précédentes le laissaient croire :

| Mécanisme appliqué | TTRSR via AttnScore | TTRSR via HiddenState |
|---|---|---|
| Noise | 87,14 % | 40,0 % |
| Noise + KeyMat | 87,14 % (inchangé) | 0,82 % (chute nette) |
| Noise + KeyMat + Head&BlockPerm | 0,0 % | 0,0 % |

**Mécanisme réel d'ISA** (texte exact, Appendix D.1, page 27) : *« The attack
optimizes input embeddings by leveraging the loss derived from hidden states.
Specifically, the attacker first records the hidden states State1 when
clients request inference using their private input X1. Subsequently,
attackers randomly initialize X2 and feeds it into the pretrained model to
evaluate State2. The attacker can use State1 and State2 to evaluate the loss,
thereby optimizing X2 to recover X1. »*

Ce n'est **pas** un modèle d'inversion pré-entraîné qu'on applique ensuite à
n'importe quelle requête (cette caractérisation, retenue dans une version
précédente de ce document, confondait ISA avec IMA — une attaque différente,
voir plus bas). C'est une **attaque par optimisation, propre à chaque
requête** : l'attaquant capture l'état interne réel d'une requête (State1),
puis fait de la descente de gradient sur un candidat X2 jusqu'à ce que son
état interne (State2) s'approche de State1 — récupérant ainsi X1.

**Pourquoi la défense marche sur les hidden states mais pas sur l'attention**
(texte exact) : *« model weights are perturbed with noise and transformation,
so that all hidden states during the forward computation process are also
noisy and transformed. Consequently, the recovered input data through ISA
would differ significantly from the original model input »* — la descente de
gradient contre un State1 bruité ne reconverge plus vers l'entrée réelle,
d'où la chute 40,0 → 0,82 dans le Tableau 4. Les scores d'attention, eux, ne
sont ni bruités ni transformés par cette protection (citation déjà notée plus
haut), donc la même technique d'optimisation appliquée à eux plutôt qu'aux
hidden states continue de fonctionner (87,14 % inchangé).

**Conséquence sur la rotation, avec la bonne justification cette fois** :
cette attaque n'a besoin ni de trafic accumulé, ni de modèle public de
référence, ni de calibration préalable — seulement de l'état interne d'**une**
requête capturée et des poids du modèle tels qu'ils étaient actifs à ce
moment précis. La rotation ne protège donc pas rétroactivement une requête
déjà capturée : elle reste attaquable par optimisation avec les poids qui
étaient en service à l'instant où elle a été traitée, indépendamment de ce
qui se passe ensuite. **La rotation n'offre donc pas de protection contre
ISA** — conclusion qui se stabilise, mais reconstruite ici sur le mécanisme
réel de l'attaque plutôt que sur des hypothèses successives invalidées.

**Deux attaques supplémentaires trouvées en lisant l'Appendix D.1**,
distinctes d'ISA — des attaques algébriques directes sur les matrices clés,
pas sur le trafic ni sur les états observés :

- **Attn-IA** : exploite un invariant mathématique dans les blocs de
  Q_query/K_key alignés sur les blocs RoPE pour retrouver directement la
  permutation secrète Π, en itérant sur les embeddings pour vérifier
  l'invariant.
- **Gate-IA** : exploite une propriété de moyenne invariante des poids de
  gate (`Avg(eW_gate) = Avg(ẽW_gate)`) pour reconstruire la relation de
  correspondance des tokens.

Les deux sont explicitement citées comme raisons supplémentaires d'introduire
la permutation tête/bloc de l'attention (elle casse la cohérence par bloc
nécessaire à ces invariants). L'obfuscation d'attention protège donc contre
**au moins trois mécanismes distincts** (ISA via AttnScore, Attn-IA, Gate-IA),
pas un seul.

**Inversion Model Attack (IMA)** — à ne plus confondre avec ISA. Texte exact :
*« With knowledge of the obfuscation mechanism, the attacker can train a
model for embedding inversion. During the training process, the attacker
iterates over a public training dataset and generates obfuscated embeddings
using the target obfuscation mechanism. »* Contrairement à ISA, IMA **est**
une attaque entraînée — mais l'attaquant génère lui-même des paires
d'entraînement en appliquant le **mécanisme d'obfuscation** (public, publié
dans le papier) avec ses propres clés, pas nécessairement la clé secrète de
la cible. Si cette lecture est correcte, IMA serait, contrairement à ISA,
transférable entre rotations (le mécanisme est public, la clé spécifique
n'a pas besoin d'être connue à l'entraînement) — mais IMA n'apparaît pas dans
le Tableau 4 et ses chiffres de TTRSR n'ont pas été localisés dans ce qui a
été lu jusqu'ici.

## Pourquoi cette distinction compte pour Secretarius

Deux cas d'usage bien différents :

1. **Protéger un gros volume de documents/conversations dans le temps** contre
   un observateur qui accumule du trafic → TFMA/SDA est la menace pertinente,
   et rotation + permutation embedding (sans attention) pourrait suffire.

2. **Protéger le contenu d'une question précise, posée une fois**, contre
   l'opérateur du serveur qui la traite → ISA est la menace pertinente, et
   son canal principal (scores d'attention) n'est **pas affecté** par la
   permutation/les clés d'embedding (citation du papier ci-dessus). La
   rotation de l'embedding n'a donc pas de prise sur ce canal. À ce jour,
   **l'obfuscation d'attention est la seule protection mesurée par le papier**
   (0 % au Tableau 4) pour ce cas d'usage précis.

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

Si le POC est concluant sur qualité et vitesse, la suite pour protéger le cas
d'usage « question isolée confidentielle » (ISA) est **l'obfuscation
d'attention** — la rotation de clé/permutation sur l'embedding n'a pas de
prise sur le canal que ISA exploite (scores d'attention), donc elle ne s'y
substitue pas. La rotation reste une défense pertinente, mais uniquement pour
le cas d'usage distinct « gros volume de trafic accumulé » (TFMA/SDA, point 1
ci-dessus) — les deux protections répondent à des menaces différentes, pas à
la même, et ne sont pas interchangeables.

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
