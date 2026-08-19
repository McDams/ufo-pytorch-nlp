## Phase 0 — Refaire les calculs du disparu

### Premier essai avec `datetime`

J'ai d'abord utilisé `datetime`, qui représente la date et l'heure auxquelles
le témoin déclare avoir vu le phénomène. Cette date semble naturelle pour
étudier les observations, mais elle ne reproduit pas les chiffres du dossier
laissé par l'analyste disparu.

Avec `datetime`, les données s'étendent du 11 novembre 1906 au 8 mai 2014,
soit 39 261 jours. Parmi les 87 459 relevés possédant une date d'observation
convertible, la moyenne est de 2,23 relevés par jour. 

| Indicateur calculé avec `datetime` | Valeur |
|---|---:|
| Période couverte | 1906-11-11 à 2014-05-08 |
| Nombre de jours | 39 261 |
| Nombre de relevés datés | 87 459 |
| Moyenne de relevés par jour | 2,23 |
| Part du samedi | 17,44 % |
| Part du lundi | 12,59 % |
| Part de juillet | 11,71 % |
| Part de février | 5,84 % |

Le samedi représente bien la journée la plus fréquente et juillet le mois le
plus fréquent. Cependant, la durée couverte et la moyenne quotidienne ne
correspondent pas aux valeurs annoncées dans le dossier. 

### Pourquoi ce premier choix est insuffisant

Le 4 juillet ne représente pas une seule journée dans l'ensemble du fichier :
il existe un 4 juillet par année. Mon premier calcul additionnait donc tous les
relevés observés un 4 juillet, soit 1 271 relevés. Cette valeur ne répond pas à
la question du dossier, qui parle d'un nombre moyen ou d'un regroupement par
date de publication. 

De plus, la journée la plus chargée avec `datetime` est le 4 juillet 2010 avec
206 relevés. Les autres journées très chargées incluent le 16 novembre 1999
avec 195 relevés et plusieurs autres 4 juillet. 

### Décision corrigée

Je retiens finalement `date_posted` pour reproduire les calculs du dossier.
Cette date correspond au moment où le Bureau a publié ou enregistré le relevé.
Elle concentre les données entre 1990 et 2014 et doit permettre de retrouver
les 8 894 jours, la moyenne de 9,2 relevés par jour et les 51 relevés associés
au 4 juillet annoncés par l'analyste disparu.

La courbe annuelle obtenue avec `datetime` ne montre pas une croissance continue
jusqu'à la fin : le volume baisse notamment en 2013 et en 2014. Cette baisse
est cohérente avec une date d'observation où les dernières années sont
incomplètes, mais elle ne reproduit pas le raisonnement du dossier. 

## Phase 1 — Le chiffre était vrai, la flotte est perdue

### 1. Ce que disait réellement le chiffre du 4 juillet

Le chiffre du 4 juillet mesure uniquement un volume de signalements : il indique
qu'un grand nombre de personnes ont envoyé un relevé ce jour-là. Il ne permet pas
de savoir ce qu'elles ont vu, si leurs observations se ressemblent, si elles sont
inquiétantes, ou si elles correspondent à une même apparition observée par de
nombreux témoins.

L'interprétation du dossier est possible, mais elle n'est pas la seule. Un pic
peut aussi être lié aux feux d'artifice et à des lumières inhabituelles dans le
ciel, à une activité collective plus importante lors d'un jour férié, ou à un
effet de déclaration : les témoins peuvent être davantage disponibles et
motivés pour écrire un signalement ce jour-là. Le même volume de relevés peut
donc recouvrir des contenus complètement différents.

Un comptage répond à la question « combien de personnes ont signalé quelque
chose ce jour-là ? ». Il ne répond pas à la question « qu'ont-elles décrit,
quelle forme ont-elles observée et quels indices ont-elles donnés dans leur
témoignage ? ».

### 2. Trois témoignages que le comptage ne voit pas

**Relevé 1 — Boulder, États-Unis, 3 octobre 2005, forme `chevron`**

> Low flying&#44 silent&#44 muted white lights on a chevron shaped glider

Ce texte apporte des informations absentes d'un comptage : l'objet est décrit
comme volant bas, silencieux, composé de lumières blanches atténuées et ayant
une forme de chevron.

**Relevé 2 — Missoula, États-Unis, 17 octobre 2003, forme `circle`**

> there was a haze around object

Ce témoignage est très court, mais il décrit une brume ou un halo autour de
l'objet. Un simple nombre de signalements ne permet pas de distinguer ce type
d'indice visuel d'une observation de lumières, d'un bruit ou d'une forme
géométrique.

**Relevé 3 — Jefferson City, États-Unis, 16 octobre 2000, forme `triangle`**

> Silent triangle object&#44 very low&#44 moving north then east.

Ce relevé indique une forme triangulaire, l'absence de bruit, une faible
altitude et une trajectoire orientée d'abord vers le nord puis vers l'est. Ces
détails permettent de caractériser l'observation, alors qu'un comptage ne
conserve que le fait qu'un relevé a été envoyé.

### 3. Commande passée au Conseil

La tâche retenue est une classification supervisée de texte : **à partir du
commentaire écrit par un témoin dans la colonne `comments`, le système doit
prédire la forme observée dans la colonne `shape`.**

- Entrée : le texte brut du témoignage.
- Sortie : une classe de forme, par exemple `light`, `triangle`, `circle` ou
  `fireball`.

Le jeu contient 88 679 relevés chargés, dont 35 commentaires vides. La colonne
`shape` contient 2 922 valeurs manquantes et 29 formes distinctes non vides.
Les classes les plus fréquentes sont `light` avec 17 872 relevés, `triangle`
avec 8 489 relevés et `circle` avec 8 453 relevés. 

## Phase 2 — Le test d'acceptation du Bureau

### Objectif du test

Avant d'entraîner un modèle sur l'ensemble de la transmission, un test
d'acceptation a été réalisé sur huit relevés réels. Le modèle reçoit un
témoignage textuel et doit prédire sa forme observée.

Le test ne cherche pas à mesurer la généralisation. Les huit relevés utilisés
pour l'entraînement sont aussi ceux utilisés pour l'évaluation. Son unique but
est de vérifier que le montage PyTorch peut apprendre : préparation du texte,
vocabulaire, encodage des classes, passage avant, calcul de la perte,
rétropropagation, mise à jour des poids et prédiction finale.

### Montage utilisé

Le modèle est composé :

- d'une couche `EmbeddingBag` qui transforme les mots du témoignage en vecteurs
  puis calcule une représentation moyenne du texte ;
- d'une couche linéaire qui produit un score pour chaque forme ;
- d'une fonction de perte `CrossEntropyLoss` ;
- d'un optimiseur Adam avec un taux d'apprentissage de 0,05.

Les huit formes retenues sont : `cigar`, `circle`, `disk`, `fireball`, `light`,
`oval`, `sphere` et `triangle`. 

### Résultat du test

| Indicateur | Valeur |
|---|---:|
| Nombre de relevés | 8 |
| Nombre de classes | 8 |
| Nombre d'itérations nécessaires | 2 |
| Perte initiale | 2,049 |
| Perte finale | 1,269 |
| Prédictions correctes finales | 8 / 8 |
| Test accepté | Oui |

La perte diminue de 2,049 à 1,269 en deux itérations. Les huit prédictions sont
correctes : le montage a donc réussi à mémoriser les huit exemples soumis.

### Prédictions finales

| Forme réelle | Forme prédite | Résultat |
|---|---|---|
| `cigar` | `cigar` | Correct |
| `circle` | `circle` | Correct |
| `disk` | `disk` | Correct |
| `fireball` | `fireball` | Correct |
| `light` | `light` | Correct |
| `oval` | `oval` | Correct |
| `sphere` | `sphere` | Correct |
| `triangle` | `triangle` | Correct |

Les huit exemples utilisés sont des témoignages réels de la transmission. Les
probabilités associées à la classe prédite ne sont pas encore toutes élevées,
mais la classe ayant le score maximal est correcte pour chaque relevé. Cela est
suffisant pour le test d'acceptation. 

### Interprétation

Ce test prouve que le pipeline de classification fonctionne techniquement et que
le réseau peut réduire sa perte puis mémoriser les exemples qu'il reçoit.

En revanche, il ne prouve pas que le modèle reconnaîtra correctement la forme
de nouveaux témoignages. Il n'y a ici ni découpe train/test, ni validation, ni
mesure honnête de généralisation. Ces éléments seront introduits à la phase 3.

## Phase 3 — Battre le service statistique

### Décisions de nettoyage

- Les relevés sans forme (`shape` vide) sont exclus : 2 922 relevés.
- Les deux fourre-tout `unknown` et `other` sont exclus, ainsi que les relevés
  au commentaire vide : 12 566 relevés au total pour ce motif et les vides.
- `round` est fusionné dans `circle`, `changed` est fusionné dans `changing`.
- Les classes de moins de 5 relevés sont exclues (`crescent`, `pyramid`,
  `flare`, `hexagon`, `dome`) : la validation stratifiée ne peut pas répartir
  des classes aussi rares de façon fiable.

| Indicateur | Valeur |
|---|---:|
| Relevés totaux chargés | 88 679 |
| Relevés gardés après nettoyage | 73 177 |
| Classes retenues | 20 |
| Taille train / validation | 58 541 / 14 636 |

### Trois modèles, même découpe

| Modèle | Accuracy validation | Temps d'entraînement |
|---|---:|---:|
| Baseline majoritaire (`light`) | 24,42 % | — |
| TF-IDF + régression logistique | 49,11 % | 32,8 s |
| PyTorch `EmbeddingBag` + MLP | **53,14 %** | 133,3 s (7 époques) |

Le modèle PyTorch dépasse le linéaire de 4 points. Le passage du texte brut au
premier nombre qui entre dans le réseau tient en trois étapes : tokenisation
par expression régulière (`[a-z0-9]+`, minuscules), conversion en identifiants
de vocabulaire (construit sur le train uniquement, `<UNK>` pour les mots
inconnus en validation), puis moyenne des embeddings de tous les jetons du
relevé via `EmbeddingBag(mode="mean")`. Objectif atteint : validé.

## Phase 4 — Le carnet de pannes

Trois pannes fabriquées sur le montage de la phase 3, une à la fois, remises
d'aplomb entre chaque essai.

| # | Panne | Geste exact | Signature | Test (< 1 min) |
|---|---|---|---|---|
| 1 | Excellent à l'entraînement, bête à l'évaluation | Ne jamais appeler `modele.eval()` : le Dropout(0,30) reste actif en validation | Perte de validation plus haute et instable, perte d'entraînement inchangée | Repasser deux fois le même lot sans rien changer : sorties différentes ⇒ mode entraînement toujours actif |
| 2 | Perte propre, prédictions pires que le hasard | `logits.argmin()` au lieu de `logits.argmax()` pour décoder la prédiction | Courbes de perte normales, accuracy tombée à 0,01 % (hasard pur : 5 %) | Comparer l'accuracy rapportée au seuil 1/nombre de classes |
| 3 | La perte se fige | Optimiseur créé avec `lr=0.0` | Perte quasi constante malgré des gradients non nuls | Comparer les poids avant/après un `step()` : identiques |

Une variante plus radicale de la panne 3 (geler tous les paramètres avec
`requires_grad=False`) a d'abord été essayée : elle ne fige pas la perte,
elle fait planter `backward()` — un échec bruyant, pas silencieux. Le taux
d'apprentissage nul est plus fidèle au symptôme demandé et beaucoup plus
proche d'un vrai bug de configuration.

La grille de diagnostic finale confirme qu'un seul test s'allume par panne
(diagonale propre) : chaque test distingue bien sa panne des deux autres en
moins d'une minute.

## Phase 5 — Le budget de calcul

Trois réglages testés un par un sur la configuration de référence de la
phase 3 (reproduite à neuf sur cette machine pour une comparaison honnête),
avant d'être combinés :

| Réglage | Temps | Gain vs référence | Écart d'accuracy |
|---|---:|---:|---:|
| Référence (phase 3, rejouée) | 156,5 s | ×1,00 | — |
| A — cache de tokenisation | 144,9 s | ×1,08 | 0,00 pt |
| B — lot de 512 au lieu de 128 | 63,4 s | ×2,47 | −0,23 pt |
| C — `OneCycleLR`, pic à 0,01, 5 époques | 76,1 s | ×2,06 | −0,25 pt |
| **Combinaison finale (A+B+C)** | **24,7 s** | **×6,34** | **−0,01 pt** |

Le réglage A (mise en cache de la tokenisation, qui évite de refaire tourner
la regex du tokenizer à chaque époque pour chaque exemple) ne coûte rien. B et
C coûtent chacun un peu moins d'un quart de point pris isolément, mais leur
combinaison ne perd quasiment rien (accuracy 53,57 % contre 53,59 % pour la
référence) tout en étant 6,3 fois plus rapide. Objectif atteint : score non
inférieur à la phase 3, facteur de gain net.

## Phase 6 — Le champ de vision du modèle

### Montage

`EmbeddingBag` (phase 3) fait déjà la moyenne de tous les jetons — son champ
de vision est trivialement total, mais il perd toute notion d'ordre. Pour
cette phase, un modèle positionnel a été construit : `Embedding` par jeton,
puis 5 blocs de convolution 1D dilatée (`dilation` = 1, 2, 4, 8, 16, noyau 3)
avec connexion résiduelle et `BatchNorm1d`, aucune récurrence — toutes les
positions sont traitées de front.

### Champ récepteur, calculé avant tout entraînement

| Couche | Dilatation | Portée cumulée (1 côté) | Diamètre cumulé |
|---:|---:|---:|---:|
| 1 | 1 | 1 | 3 |
| 2 | 2 | 3 | 7 |
| 3 | 4 | 7 | 15 |
| 4 | 8 | 15 | 31 |
| 5 | 16 | 31 | 63 |

Longueur maximale acceptée : 55 jetons (médiane 14). Le **diamètre** total
(63) dépasse la longueur maximale (55) : une position centrale peut voir tout
le relevé. Mais la **portée** d'un seul côté (31) ne suffit pas à couvrir
55 − 1 = 54 positions : une position en bord de séquence ne voit pas l'autre
bord à elle seule.

### Vérification expérimentale (avant entraînement)

En perturbant le premier jeton du relevé le plus long : la dernière position
du tronc convolutif ne bouge pas (écart 0,0 — cohérent avec une portée de 31
contre une distance de 54), et la dernière position réellement influencée est
exactement la position 31, comme prédit par le calcul. Mais la sortie
réellement utilisée par le classifieur — après moyenne masquée sur toutes les
positions — bouge bien (écart de 0,010 sur les logits) : le premier mot
influence les positions 0 à 31 du tronc, qui entrent toutes dans cette
moyenne. C'est le pooling, pas une position isolée, qui garantit la
dépendance à l'ensemble du relevé.

### Score

| Modèle | Accuracy |
|---|---:|
| Phase 3 (EmbeddingBag + MLP) | 53,14 % |
| Phase 6 (convolutif dilaté) | **54,98 %** |

Le modèle convolutif dépasse la phase 3 de près de 2 points, en 1 616 s
(9 époques, jeu complet).

## Phase 7 — Quatre relevés à la fois

### Budget de calcul

Un lot de 4 sur les 58 541 exemples complets aurait coûté environ 19 minutes
par époque (mesuré : 77 ms/itération × 14 636 itérations). Décision prise
avant toute mesure : les essais de cette phase tournent sur un sous-échantillon
stratifié fixe de 3 000 relevés d'entraînement / 900 de validation, identique
pour les quatre essais — seule la taille de lot varie.

| Essai | Accuracy | Instabilité* |
|---|---:|---:|
| `BatchNorm1d`, lot = 4 (avant) | 41,0 % | 0,058 |
| `BatchNorm1d`, lot = 512 (repère) | 39,2 % | 0,068 |
| `GroupNorm`, lot = 4 (après) | 40,4 % | 0,077 |
| `GroupNorm`, lot = 512 (après, gros lot) | 37,0 % | 0,017 |

*écart-type des variations de perte de validation d'une époque à l'autre.

### Résultat honnête, pas celui attendu

Contrairement au symptôme classique attendu, `BatchNorm1d` au lot de 4 n'est
ici ni moins précis ni moins stable que la référence au lot de 512, et
`GroupNorm` ne montre aucun gain net mesurable sur ce montage. Raison
identifiée : `BatchNorm1d` sur un tenseur `(batch, canaux, longueur)` agrège
le lot **et** les positions de la séquence pour calculer ses statistiques —
avec `longueur = 55`, un lot de 4 fournit encore environ 220 valeurs par
canal, largement assez pour rester stable. Le cas franchement pathologique de
la littérature correspondrait à un `BatchNorm1d` posé directement sur le
vecteur poolé `(batch, canaux)`, ce que ce montage ne fait pas.

Ce que `BatchNorm1d` couplait entre les relevés du même lot reste vrai par
construction (ses statistiques sont partagées entre tous les exemples du
lot), même si l'effet ne saute pas aux yeux ici. `GroupNorm` retire ce
couplage architecturalement, sans coût mesuré. À l'évaluation, `eval()`
utilise les statistiques glissantes (leçon de la phase 4), donc prédire sur
un relevé unique fonctionne déjà avec `BatchNorm1d` — le vrai risque se
situerait à l'**entraînement** avec un lot de taille 1, un cas non testé ici.

## Phase 8 — Le Conseil a lu trois relevés

### Liste des mots interdits (46 mots)

Les 20 formes retenues, leurs pluriels, les deux doublons produits par la
fusion de la phase 3 (`round`, `changed`, et leurs pluriels), et la variante
d'écriture `disc`/`discs` pour `disk`. Liste complète dans
`outputs/phase_8_vocabulaire_interdit/liste_mots_interdits.csv`.

L'interdiction est appliquée **au niveau des jetons** (le même tokenizer que
le modèle), pas par une regex `\bmot\b` sur le texte brut : un premier essai
par regex laissait passer `"...ball of light_w/aura..."`, où `_` compte comme
caractère de mot pour `\b` mais pas pour le tokenizer `[a-z0-9]+` — les deux
définitions de « mot » n'étaient pas les mêmes. Après correction, le compte de
relevés contenant encore un mot interdit est **0**, vérifié par le code.

### Repère de cohérence avec le dossier du Conseil

| Indicateur | Nos données | Conseil |
|---|---:|---:|
| Part globale (mot de la forme présent) | 40,5 % | 34,7 % |
| Part pour `light` | 71,4 % | 72,6 % |
| Part pour `circle` | 19,1 % | 9,9 % |

### La chute

| Résumé | Avant | Après | Chute |
|---|---:|---:|---:|
| Accuracy globale | 53,14 % | 37,07 % | **−16,07 pt** |
| F1 macro-moyenné | 42,47 % | 15,09 % | **−27,39 pt** |

L'accuracy globale est pondérée par la fréquence des classes (`light`,
`triangle`, `circle` dominent le total) ; le F1 macro donne le même poids à
chacune des 20 classes. Le F1 macro chute presque deux fois plus fort : une
partie du score de la phase 3 venait de classes qui s'effondrent complètement
sans leur mot-nom, en particulier `diamond` (F1 0,566 → 0), `egg` (0,461 → 0),
`cigar` (0,603 → 0,12), `teardrop` (0,396 → 0) et `cross` (0,140 → 0). `light`,
la classe majoritaire, résiste le mieux (0,612 → 0,526) : elle dispose d'assez
d'exemples pour que le modèle apprenne d'autres indices que le mot lui-même.

## Phase 9 — Rendre des comptes sur trois décisions

Attribution mot par mot par occlusion (retrait d'un mot, mesure de la chute de
probabilité de la classe prédite) sur le modèle de la phase 8, trois relevés
de la validation :

**Relevé réussi (`large saucer` → `disk`, confiance 100 %).** Le modèle a
retenu un seul mot, `saucer`, qui porte à lui seul la quasi-totalité de la
décision (score 0,99) ; `large` ne pèse rien. Ce n'est pas de la compréhension
distribuée du témoignage, c'est de la reconnaissance d'un synonyme appris —
`saucer` n'est pourtant pas dans la liste des mots interdits.

**Relevé raté (`ufo flying saucer`, vraie forme `sphere`, prédite `disk`,
confiance 99,999 %).** Le mot `saucer` déclenche le même mécanisme que dans
l'exemple réussi (score 0,74) : le modèle est cohérent avec lui-même. Une
soucoupe volante est un objet en forme de disque presque par définition ;
l'étiqueter `sphere` dans le fichier source est au moins aussi défendable que
`disk`. Ce raté apprend surtout que la colonne `shape` contient des choix
d'annotation ambigus, pas que le modèle raisonne mal.

**Relevé hésitant (vraie forme `changing`, prédictions `light` 26,5 % /
`circle` 26,5 %, quasi à égalité).** Le témoignage est technique et daté ; les
mots qui soutiennent le plus `light` sont `dots`, `white` et `beams`. Mais
rien dans le vocabulaire de surface ne porte l'idée de **changement au cours
du temps** que porte la forme réelle `changing` — un concept narratif, pas un
objet statique nommable par un mot isolé. Un modèle qui moyenne des embeddings
de mots n'a structurellement aucune prise dessus.

## Phase 10 — Chaque mot interroge les autres

Mécanisme d'attention codé à la main (tenseurs, produit matriciel, softmax,
couches linéaires — rien d'autre), sur un vrai relevé contenant une reprise :
`« Oval shaped with lights all around it in a haze with several smaller
lights flying all around it. »` (18 jetons, `it` en positions 6 et 17).

Trois vecteurs par mot (question, étiquette, contenu) via trois couches
linéaires sans biais, puis `scores = Q @ Kᵀ / √d`, `poids = softmax(scores)`,
`sortie = poids @ V`.

| Vérification | Résultat |
|---|---|
| Chaque ligne de la matrice somme à 1 | ✅ |
| La sortie a la même forme que l'entrée | ✅ (18, 32) |
| Position du mot sur lequel s'appuie le plus chaque `it` | désignée dans le notebook |

Le modèle n'est pas entraîné (poids aléatoires) : la case désignée n'a pas de
raison de correspondre à `oval`, le mot que `it` reprend réellement. Ce qui
est démontré ici, c'est que le mécanisme sait pointer une case précise pour
cette question, pas qu'il a déjà appris la bonne réponse.

## Phase 11 — Le Conseil mélange vos mots

Sur ce même relevé, permutation aléatoire des positions puis réalignement de
la sortie mélangée dans l'ordre d'origine (chaque mot remis à sa place de
départ) :

| Mesure | Avant encodage de position | Après |
|---|---:|---:|
| Écart max., sortie correcte vs sortie mélangée-réalignée | 8,9 × 10⁻⁸ (nul aux erreurs de calcul flottant près) | **0,139** |

Le mécanisme d'attention lui-même n'a pas été touché : c'est un encodage de
position sinusoïdal, ajouté aux vecteurs d'entrée **avant** qu'ils ne
deviennent question/étiquette/contenu, qui casse l'invariance à l'ordre.
Injecté avant les trois projections, il infuse les trois rôles à la fois ;
injecté après (sur la sortie déjà calculée), il n'aurait rien réparé, puisque
les poids d'attention auraient déjà été calculés sans lui.

## Phase 12 — Le Conseil demande la facture

Chronométrage de l'attention de la phase 11 (inchangée), 50 mesures par
longueur, médiane retenue :

| Longueur | Temps médian | Taille de la matrice de poids |
|---:|---:|---:|
| 32 | 0,292 ms | 1 024 cases |
| 64 | 0,355 ms | 4 096 cases |
| 128 | 0,477 ms | 16 384 cases |
| 256 | 0,569 ms | 65 536 cases |
| 512 | 2,028 ms | 262 144 cases |

La taille de la matrice quadruple exactement à chaque doublement de longueur
(propriété géométrique en longueur²). Le temps mesuré, lui, ne suit ce facteur
4 qu'au tout dernier doublement (256 → 512, ×3,57) : en dessous, le temps d'un
passage est dominé par la latence fixe de Python/PyTorch, pas par le calcul —
les doublements précédents donnent ×1,22, ×1,34 et ×1,19, loin de 4.

**Conclusion honnête** : à cette taille de modèle (`D_MODEL = 32`, une tête),
un simple passage avant ne devient jamais « inutilisable » aux longueurs
testées — l'extrapolation en longueur² donne encore moins de 500 ms à 8 000
jetons. Ce qui limiterait réellement un entraînement à cette échelle, ce
n'est pas le chronomètre d'un passage isolé, c'est ce même coût répété à
chaque exemple et chaque époque, et surtout la mémoire nécessaire pour
conserver la matrice de poids en vue de la rétropropagation. Pour cette
transmission (55 jetons au maximum), la question ne se pose de toute façon
pas.

## Phase 13 — Deux regards sur le même relevé

Deux têtes d'attention (16 dimensions chacune, initialisations différentes)
sur le même relevé, sorties concaténées puis recombinées par une couche
linéaire (32, 18) → (32, 18) inchangée en forme.

| Mesure (écart absolu moyen entre les deux matrices) | Valeur |
|---|---:|
| Têtes réelles (initialisations différentes) | 0,0205 |
| Cas de contrôle (deux têtes strictement identiques) | 0,0000 |

Le cas de contrôle confirme que le désaccord mesuré n'est pas un artefact de
la méthode : deux têtes identiques donnent un désaccord nul. Les têtes 1 et 2
ne sont pas entraînées, leur différence vient uniquement de l'initialisation
aléatoire — pas encore d'une spécialisation apprise. Avec un entraînement, on
pourrait vérifier si chaque tête se stabilise sur un type de relation
particulier (sujet-verbe, reprise pronominale) d'un relevé à l'autre.
de mots n'a structurellement aucune prise dessus.