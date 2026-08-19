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