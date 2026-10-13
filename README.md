# trustpilot-check

Notebook Colab qui évalue l'authenticité des avis d'une page Trustpilot. À partir de l'URL, il
collecte les avis et l'historique public de leurs auteurs, cherche les signaux d'avis fabriqués
et classe les avis à relire en priorité, avec ses raisons pour chacun.

Un signal n'est pas une preuve : un client furieux crée souvent son compte le jour où il écrit.
La décision reste humaine. Les mêmes contrôles portent sur les avis positifs et négatifs, ce qui
permet de les comparer.

## Utilisation

1. Ouvrir `authenticite_avis.ipynb` dans Google Colab.
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

## Développement

Le notebook est généré à partir de `src/authenticite_avis.py` (cellules au format `# %%`) :

```bash
python3 outils/construire_notebook.py
```
