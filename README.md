# trustpilot-check : détecter les faux avis Trustpilot

[![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kofekod23/trustpilot-check/blob/main/authenticite_avis.ipynb)

Notebook Google Colab qui vérifie l'authenticité des avis d'une page Trustpilot. À partir de
l'URL, il collecte les avis et l'historique public de leurs auteurs, cherche les signes de faux
avis (positifs comme négatifs) et classe les avis à relire en priorité, avec ses raisons pour
chacun : rafales d'avis, comptes créés pour l'occasion, textes recopiés ou paraphrasés, style
d'écriture, fautes d'orthographe, sentiment contradictoire avec la note, société signataire
fermée au registre.

Un signal n'est pas une preuve : un client furieux crée souvent son compte le jour où il écrit.
La décision reste humaine. Les mêmes contrôles portent sur les avis positifs et négatifs, ce qui
permet de les comparer.

## Utilisation

1. Ouvrir le notebook avec le bouton « Open in Colab » ci-dessus.
2. Facultatif : ajouter la clé `TYPESAFE_API_KEY` dans les secrets Colab pour les jugements de
   Jev. Sans clé, le reste fonctionne.
3. Saisir l'URL Trustpilot dans « Paramètres », puis Exécution > Tout exécuter.

Si Trustpilot bloque le navigateur de Colab, cocher `IMPORTER_PAGES_HTML` et téléverser les
pages d'avis enregistrées depuis son propre navigateur (Ctrl+S).

## Ce qui est mesuré

- **Temps** : rafales d'avis, comptes créés le jour de l'avis, réveil après un long silence.
- **Auteurs** : avis unique, avis en série, nom générique ou de société, société fermée au
  registre.
- **Textes** : copies et paraphrases entre auteurs, gabarit générique, style d'IA, promotion
  d'un concurrent, sentiment du texte comparé à la note.
- **Langue** : fautes, ponctuation, habitudes d'écriture, expressions reprises par plusieurs
  comptes.

Les signaux sont combinés en un indice de vigilance de 0 à 10, dont les poids et les seuils se
règlent dans le notebook. Les résultats sont exportés en Excel, et sur Google Drive si l'option
est cochée.

## Modèles

`bge-m3` pour les paraphrases, `distilcamembert` (français) et `nlptown` (autres langues) pour
le sentiment, Jev pour les jugements sur le contenu, LanguageTool pour la grammaire. Le détail
de chaque modèle, de ses seuils et de leur calibration est dans [docs/modeles.md](docs/modeles.md).

Le premier lancement télécharge environ 5 Go de modèles ; le GPU de Colab accélère les grandes
pages.

## Limites

- Seuls les avis visibles publiquement sont analysés, et l'historique d'un auteur se limite à
  ses 20 derniers avis.
- Sur une page qui compte peu d'avis, les tests statistiques ne repèrent que des anomalies
  grossières.
- Playwright est épinglé en version 1.62.0 : la 1.63 plante sur les pages Trustpilot.

## In English

**trustpilot-check** is a Google Colab notebook for fake review detection on Trustpilot. Paste a
Trustpilot company URL and it scrapes the reviews and the reviewers' public history, then scores
each review on authenticity signals: review bursts (Poisson test), throwaway accounts,
copy-paste and paraphrase detection across authors (TF-IDF, `bge-m3` embeddings), writing style
and spelling, sentiment versus star rating, competitor promotion, and company names checked
against the French business registry. It outputs a 0 to 10 suspicion index with reasons, charts
and an Excel export. The interface and comments are in French; reviews in any language can be
analysed.

## Développement

Le notebook est généré à partir de `src/authenticite_avis.py` (cellules au format `# %%`) :

```bash
python3 outils/construire_notebook.py
```
