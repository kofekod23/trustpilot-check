# Modèles et méthodes

Ce document décrit chaque modèle utilisé par le notebook, ce qu'il mesure, comment son résultat
entre dans l'indice de vigilance et comment ses seuils ont été fixés.

Les seuils ont été calibrés sur une page Trustpilot réelle (53 avis, octobre 2026). Sur
une page très différente (autre langue, autre secteur, avis beaucoup plus longs), il vaut la
peine de les vérifier.

## Vue d'ensemble

| Étape | Outil | Où il tourne | Coût |
|---|---|---|---|
| Paraphrases entre auteurs | `BAAI/bge-m3` | Colab (CPU ou GPU) | gratuit, environ 4 Go à télécharger |
| Copies et habitudes d'écriture | TF-IDF (scikit-learn) | Colab | gratuit |
| Sentiment des avis en français | `cmarkea/distilcamembert-base-sentiment` | Colab | gratuit, 260 Mo |
| Sentiment des autres avis | `nlptown/bert-base-multilingual-uncased-sentiment` | Colab | gratuit, 640 Mo |
| Jugements sur le contenu | Jev (TypeSafe) | API | environ 0,003 $ pour 50 avis |
| Orthographe | pyspellchecker (dictionnaires français et anglais) | Colab | gratuit |
| Grammaire et typographie | LanguageTool | API publique | gratuit, 20 requêtes par minute |
| Vocabulaire et expressions | log-odds, n-grammes | Colab | gratuit |
| Rafales | loi de Poisson | Colab | gratuit |
| Sociétés signataires | API recherche-entreprises, BODACC | API publiques | gratuit |

## Paraphrases : `BAAI/bge-m3`

Une campagne de faux avis recopie rarement le même texte : elle le reformule. Pour repérer deux
avis qui disent la même chose avec d'autres mots, le notebook transforme chaque texte en vecteur
avec `bge-m3`, un modèle d'embeddings multilingue (568 M de paramètres) qui fonctionne bien en
français. La ressemblance entre deux avis est le cosinus entre leurs vecteurs normalisés.

Pour chaque avis, le notebook garde la ressemblance la plus forte avec un avis d'un **autre**
auteur (`sim_max_sens`). Au-delà de 0,80 (`SEUILS["paraphrase_semantique"]`), le signal
`paraphrase_autre_auteur` s'allume (poids 1,5).

**Calibration.** Huit vrais avis négatifs ont été reformulés à la main (mêmes faits, autres
mots). Chaque modèle devait séparer ces 8 paraphrases des 1 378 paires de vrais avis
d'auteurs différents, qui ne contiennent aucune copie.

| Modèle | Avis distincts : médiane / maximum | Paraphrases : minimum / médiane | Paraphrases repérées sans fausse alerte |
|---|---|---|---|
| `paraphrase-multilingual-MiniLM-L12-v2` | 0,29 / 0,77 | 0,72 / 0,82 | non : les deux zones se chevauchent |
| `BAAI/bge-m3` (retenu) | 0,48 / 0,75 | 0,83 / 0,86 | 8 sur 8 entre 0,76 et 0,83 |
| `intfloat/multilingual-e5-large` | 0,85 / 0,93 | 0,91 / 0,96 | non : les deux zones se chevauchent |

L'ancien modèle, MiniLM, ne repérait que 2 paraphrases sur 8 au seuil de 0,85. `e5-large` donne
0,85 en médiane à des avis sans rapport et monte jusqu'à 0,93, ce qui rend tout seuil fragile. `bge-m3` est le
seul à laisser un écart net entre les deux groupes ; le seuil de 0,80 est placé au milieu.

**Temps de calcul.** Le premier lancement télécharge environ 4 Go. Ensuite, il faut compter
environ 45 secondes pour 50 avis sur un processeur, chargement compris. Le
GPU de Colab (Exécution > Modifier le type d'exécution) accélère nettement les pages de
plusieurs centaines d'avis.

## Copies et habitudes d'écriture : TF-IDF

Trois mesures complètent les embeddings, toutes par cosinus entre vecteurs TF-IDF :

- **caractères** (n-grammes de 3 à 5 caractères) : repère les copies, même avec quelques mots
  changés ou des fautes ajoutées. Au-delà de 0,60, le signal `texte_recopie` s'allume (poids 3,
  le plus fort de l'indice) ;
- **mots et paires de mots** : ressemblance lexicale, affichée dans le tableau des paires ;
- **mots-outils** (le, de, je, mais, très…) : leur fréquence trace une manière d'écrire plus que
  le sujet. Avec les **habitudes typographiques** (espace avant le point d'exclamation,
  apostrophe droite ou courbe, points de suspension, majuscules, longueur des phrases), elle
  aide à soupçonner une même personne derrière plusieurs comptes.

Les deux dernières mesures sont affichées pour la relecture mais n'entrent pas dans l'indice :
sur des textes de quelques dizaines de mots, elles ne suffisent pas à rattacher deux comptes.

## Sentiment comparé à la note

Un avis à une étoile au texte élogieux, ou l'inverse, peut trahir une erreur, de l'ironie ou un
avis écrit pour gonfler une note. Le notebook lit donc la note « exprimée » par le texte et la
compare à la note donnée.

**Deux modèles selon la langue.**

- Avis en français : `cmarkea/distilcamembert-base-sentiment`, un CamemBERT allégé entraîné sur
  des avis français notés de 1 à 5 étoiles.
- Autres langues : `nlptown/bert-base-multilingual-uncased-sentiment`, entraîné sur des avis
  de produits en six langues, également notés de 1 à 5.

**Note attendue.** Plutôt que de garder l'étoile la plus probable, le notebook calcule la
moyenne des étoiles pondérée par les probabilités du modèle (`sentiment_modele_etoiles`, de 1
à 5). Un texte neutre qui hésite entre 1 et 5 étoiles obtient ainsi une note proche de 3 au lieu
de basculer d'un côté.

**Calibration** sur les 53 avis, en note attendue :

| Combinaison | Écart moyen à la note | Écart moyen, avis français | Avis à 2 étoiles d'écart ou plus | Avis négatifs lus comme positifs |
|---|---|---|---|---|
| `nlptown` seul (ancien) | 0,52 | 0,50 | 1 | 1 |
| `distilcamembert` seul | 0,59 | 0,45 | 5 | 3 |
| `distilcamembert` en français, `nlptown` ailleurs (retenu) | 0,48 | 0,45 | 2 | 0 |

En gardant l'étoile la plus probable, `distilcamembert` trouve la note exacte de 86 % des avis
français, contre 80 % pour `nlptown`,
mais il prend des avis anglais négatifs pour des avis positifs, d'où le partage par langue.

**Utilisation dans l'indice.** Le signal `incoherence_note_texte` (poids 2) s'allume si le
sentiment jugé par Jev s'écarte de 2 niveaux ou plus de la note, ou si la note attendue du
modèle local s'écarte d'au moins 2,5 étoiles (`SEUILS["ecart_sentiment_modele"]`) sans que Jev
le contredise. Ce garde-fou vient d'un cas réel : un relevé de prix sans aucun mot négatif
(« prix courant 58 €, prix chez eux 96 € ») était lu comme un avis à 5 étoiles par l'ancien
calcul.

## Jugements sur le contenu : Jev

Jev est un modèle de TypeSafe qui répond à des questions précises par des probabilités, des
scores ou un choix, plutôt que par du texte. Le notebook lui pose les questions suivantes pour
chaque avis, en un seul appel :

| Question | Type | Ce qu'elle mesure |
|---|---|---|
| `experience_precise` | oui/non | un fait situé : agence, produit, montant, échange, chronologie |
| `plainte_generique` | oui/non | un texte interchangeable, valable pour n'importe quelle entreprise |
| `style_ia` | oui/non | formules toutes faites, structure trop régulière, aucune faute ni oralité |
| `invraisemblance` | oui/non | contradiction interne ou récit impossible |
| `promotion_concurrent` | oui/non | l'avis sert surtout à envoyer vers un concurrent nommé |
| `nom_entreprise` | oui/non | le nom affiché est celui d'une société (déclenche la vérification au registre) |
| `sentiment` | score sur 5 niveaux | de très négatif à très positif, indépendamment de la note |
| `qualite_informative` | score sur 4 niveaux | utilité de l'avis pour un futur client |
| `profil_auteur` | choix | professionnel, particulier ou indéterminable |
| `theme_*` | oui/non | un par thème de `THEMES` (prix, accueil, retours, livraison, site, facturation) |

**Témoins.** Avant de juger les avis, le notebook soumet sept avis inventés dont la réponse est
connue : un récit précis, une plainte générique signée par une société, deux textes au style
d'IA, un texte oral plein de fautes, une promotion de concurrent, un récit impossible. Si un
témoin échoue, la question concernée est marquée « à prendre avec réserve », et pour
`style_ia` le signal est désactivé.

**Seuil.** Une réponse oui/non compte dans l'indice à partir de 0,70 (`SEUILS["jev"]`).
`generique_sans_fait` exige en plus que `experience_precise` reste sous 0,30.

**Limites.** Deux exécutions successives peuvent donner des probabilités légèrement
différentes : un avis proche du seuil peut gagner ou perdre un signal d'un jour à l'autre. Les
thèmes n'ont pas de témoins.

Sans clé `TYPESAFE_API_KEY`, ces colonnes sont absentes ; l'indice est alors calculé sur les
signaux restants.

## Orthographe et grammaire

- **pyspellchecker** compte les mots absents du dictionnaire français ou anglais
  (`taux_mots_inconnus`), noms propres et mots en majuscule exclus.
- **LanguageTool** (API publique, une requête toutes les 3,2 secondes) classe les erreurs en
  orthographe, grammaire et ponctuation ou typographie, puis les ramène à 100 mots
  (`lt_erreurs_pour_100_mots`). `MAX_TEXTES_LANGUAGETOOL` limite le nombre d'avis envoyés,
  négatifs en premier.

Ces mesures servent à comparer les groupes (négatifs et positifs) et n'entrent pas dans
l'indice : un texte sans faute peut venir d'un client soigneux, et des fautes s'imitent.

## Vocabulaire et expressions

- **Mots distinctifs** : log-odds avec a priori de Dirichlet (Monroe, Colaresi et Quinn,
  2008). La méthode fait ressortir les mots propres aux avis négatifs par rapport aux autres,
  sans surestimer les mots rares.
- **Expressions partagées** : suites de 3 mots, dont au moins 2 mots pleins, reprises par au
  moins 3 auteurs différents. Une formule commune à plusieurs comptes peut trahir un modèle de
  texte fourni.

## Rafales

Les avis négatifs et les autres sont traités séparément. À partir de chaque avis, le notebook
compte les avis du même groupe publiés dans les 30 jours suivants et compare ce nombre, par une
loi de Poisson, au rythme moyen sur un an de part et d'autre, limité à la période réellement
couverte par les données. Comme on teste une fenêtre par avis, la probabilité est multipliée
par le nombre d'avis du groupe (correction de Bonferroni). Chaque avis reçoit la probabilité de
la fenêtre la plus chargée qui le contient, et le signal `dans_une_rafale` s'allume sous 0,05.

## Registre des sociétés

Quand Jev ou une règle sur le nom (SARL, SAS, « .com »…) indique qu'un auteur signe au nom
d'une société, le notebook interroge l'API recherche-entreprises puis le BODACC : statut, date
de création, procédures collectives, date de fermeture. Le signal `societe_fermee` s'allume si
la société était fermée avant la date de l'avis. Une homonymie reste possible : une
correspondance de nom ne prouve pas l'identité.

## Ce qui n'est pas utilisé

Les détecteurs automatiques de texte généré par une IA ne sont pas utilisés. Ceux qui existent
sont surtout entraînés en anglais, se trompent souvent sur des textes courts et vieillissent à
chaque nouvelle génération de modèles. La question `style_ia` de Jev, contrôlée par des
témoins, joue ce rôle.

## Changer de modèle

Les noms des modèles sont dans la source du notebook, `src/authenticite_avis.py` :
`MODELE_SENTIMENT_FR`, `MODELE_SENTIMENT_MULTILINGUE`, et l'appel à `SentenceTransformer` dans
la cellule « Ressemblance entre textes ». Après un changement de modèle d'embeddings, il faut
recalibrer `SEUILS["paraphrase_semantique"]` : chaque modèle a sa propre échelle de
similarité, comme le montre la colonne des médianes plus haut. Régénérer ensuite le notebook
avec `python3 outils/construire_notebook.py`.
