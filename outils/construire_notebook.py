"""Construit authenticite_avis.ipynb à partir de src/authenticite_avis.py (format « # %% »)."""
import json
import re
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
SOURCE = RACINE / "src" / "authenticite_avis.py"
CIBLE = RACINE / "authenticite_avis.ipynb"

# Cellules dont le code est masqué dans Colab (seul le titre et les champs restent visibles).
FORMULAIRES = {"Paramètres"}


def cellules(texte: str) -> list[dict]:
    blocs = re.split(r"^# %%", texte, flags=re.M)[1:]
    sortie = []
    for bloc in blocs:
        entete, _, corps = bloc.partition("\n")
        corps = corps.strip("\n")
        if entete.strip().startswith("[markdown]"):
            lignes = [re.sub(r"^# ?", "", l) for l in corps.splitlines()]
            sortie.append({"cell_type": "markdown", "metadata": {}, "source": "\n".join(lignes)})
            continue
        titre = entete.strip()
        if titre:
            mode = "" if titre in FORMULAIRES else ' { display-mode: "form" }'
            corps = f"#@title {titre}{mode}\n{corps}"
        sortie.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                       "outputs": [], "source": corps})
    return sortie


notebook = {
    "nbformat": 4, "nbformat_minor": 0,
    "metadata": {"colab": {"provenance": []}, "kernelspec": {"name": "python3", "display_name": "Python 3"},
                 "language_info": {"name": "python"}},
    "cells": cellules(SOURCE.read_text(encoding="utf-8")),
}
CIBLE.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"{CIBLE} : {len(notebook['cells'])} cellules")
