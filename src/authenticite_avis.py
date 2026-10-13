# %% [markdown]
# # Authenticité des avis Trustpilot
#
# À partir de l'URL d'une page Trustpilot, ce notebook collecte les avis et l'historique de
# leurs auteurs, puis cherche les **signaux** qui distinguent un avis spontané d'un avis
# fabriqué : rythme de publication, comptes créés pour l'occasion, textes recopiés ou
# paraphrasés, style d'écriture, fautes, cohérence entre la note et le texte, nom affiché,
# société signataire fermée, etc.
#
# **Ce que le notebook ne fait pas : prouver qu'un avis est faux.** Chaque signal a une
# explication innocente possible (un client furieux crée souvent son compte le jour même).
# L'indice de vigilance classe les avis à relire en priorité, avec les raisons ; la décision
# reste humaine. Les mêmes contrôles s'appliquent aux avis positifs et négatifs, ce qui
# donne un point de comparaison.
#
# **Mode d'emploi**
# 1. Facultatif : ajouter la clé `TYPESAFE_API_KEY` dans les secrets Colab (icône clé) pour
#    les jugements du modèle Jev. Sans clé, ces colonnes sont simplement absentes.
# 2. Saisir l'URL dans « Paramètres », puis Exécution > Tout exécuter.
# 3. Si Trustpilot bloque le navigateur de Colab, cocher `IMPORTER_PAGES_HTML` : enregistrer
#    depuis votre navigateur les pages d'avis (Ctrl+S, « page web complète » ou « HTML
#    uniquement ») et, si possible, les pages profil des auteurs, puis les téléverser.
#
# | Famille | Signaux |
# |---|---|
# | Temps | rafales (test de Poisson sur 30 jours), délai expérience → avis, heure, compte créé le jour de l'avis, réveil après un long silence |
# | Auteur | avis unique, entreprise seule cible, avis en série le même jour, nom générique ou de société, société fermée au registre |
# | Texte | doublons et paraphrases entre auteurs, gabarit générique, style d'IA, invraisemblance, promotion d'un concurrent |
# | Langue | fautes d'orthographe et de grammaire, ponctuation, habitudes typographiques, vocabulaire et expressions répétées |
# | Cohérence | sentiment du texte comparé à la note donnée |

# %% Installation et imports
import subprocess
import sys

EN_COLAB = "google.colab" in sys.modules
if EN_COLAB:
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", "playwright==1.62.0", "pyspellchecker",
                    "sentence-transformers", "sentencepiece", "protobuf"], check=True)
    subprocess.run([sys.executable, "-m", "playwright", "install", "--with-deps", "chromium"],
                   check=True, stdout=subprocess.DEVNULL)

import asyncio
import json
import math
import os
import random
import re
import time
import urllib.parse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from IPython.display import HTML, Markdown, display
from scipy.stats import poisson
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from spellchecker import SpellChecker

pd.set_option("display.max_colwidth", 120)
PARIS = ZoneInfo("Europe/Paris")

# %% Paramètres
URL_TRUSTPILOT = ""  #@param {type:"string"}
MAX_PAGES_AVIS = 10  #@param {type:"integer"}
COLLECTER_PROFILS = True  #@param {type:"boolean"}
MAX_PROFILS = 150  #@param {type:"integer"}
IMPORTER_PAGES_HTML = False  #@param {type:"boolean"}
JUGEMENTS_JEV = True  #@param {type:"boolean"}
SENTIMENT_LOCAL = True  #@param {type:"boolean"}
SIMILARITE_SEMANTIQUE = True  #@param {type:"boolean"}
LANGUAGETOOL = True  #@param {type:"boolean"}
MAX_TEXTES_LANGUAGETOOL = 60  #@param {type:"integer"}
VERIFIER_REGISTRE = True  #@param {type:"boolean"}
ENREGISTRER_SUR_DRIVE = False  #@param {type:"boolean"}
DOSSIER_DRIVE = "authenticite_avis"  #@param {type:"string"}

# Thèmes de plainte jugés par Jev (modifiables) : un oui/non par thème et par avis.
THEMES = {
    "prix": "le prix, les tarifs ou des frais jugés trop élevés",
    "accueil": "l'accueil, l'amabilité ou la compétence du personnel",
    "sav_retour": "un retour, un remboursement, un avoir ou le service après-vente",
    "livraison_stock": "une livraison, un délai, une rupture de stock ou une commande non honorée",
    "site_web": "le site internet, l'application ou l'inscription en ligne",
    "facturation": "la facturation, un paiement ou des frais administratifs",
}

# Poids des signaux dans l'indice de vigilance (0 à 10). Modifiables sans relancer la collecte.
POIDS = {
    "texte_recopie": 3.0,
    "paraphrase_autre_auteur": 1.5,
    "style_ia": 2.0,
    "incoherence_note_texte": 2.0,
    "promotion_concurrent": 2.0,
    "generique_sans_fait": 1.5,
    "invraisemblance": 1.0,
    "compte_jetable": 1.0,
    "dans_une_rafale": 1.0,
    "serie_meme_jour": 1.0,
    "societe_fermee": 1.0,
    "reveil_apres_long_silence": 0.5,
    "nom_generique": 0.5,
}
# paraphrase_semantique : calibré pour bge-m3 (paraphrases témoins ≥ 0,83, avis distincts ≤ 0,75 sur une page réelle de 53 avis).
# ecart_sentiment_modele : écart entre la note donnée et la note lue en espérance (1 à 5).
# relecture : indice à partir duquel la conclusion compte un avis « à relire » (deux signaux moyens, ou un fort et un faible).
SEUILS = {"jev": 0.7, "recopie_caracteres": 0.6, "paraphrase_semantique": 0.80, "ecart_sentiment_modele": 2.5,
          "silence_jours": 3 * 365, "rafale_p": 0.05, "meme_jour": 5, "relecture": 2.0}

# %% Collecte des pages Trustpilot
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36")


def analyser_url(url: str) -> tuple[str, str, dict]:
    """Renvoie (hôte, domaine de l'entreprise, paramètres de filtre) à partir d'une URL ou d'un domaine."""
    url = url.strip()
    if not url.startswith("http"):
        url = "https://fr.trustpilot.com/review/" + url
    p = urllib.parse.urlparse(url)
    m = re.match(r"/review/([^/?#]+)", p.path)
    if not m:
        raise ValueError(f"URL Trustpilot non reconnue : {url} (attendu : https://fr.trustpilot.com/review/<domaine>)")
    params = urllib.parse.parse_qs(p.query)
    params.pop("page", None)
    params.setdefault("languages", ["all"])
    return p.netloc or "fr.trustpilot.com", m.group(1), params


def url_page(hote: str, domaine: str, params: dict, num: int) -> str:
    q = dict(params, **({"page": [str(num)]} if num > 1 else {}))
    return f"https://{hote}/review/{domaine}?{urllib.parse.urlencode(q, doseq=True)}"


def lire_next_data(html: str) -> dict | None:
    m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.S)
    return json.loads(m.group(1)) if m else None


async def _recuperer(urls: list[str], accueil: str, attente_max: int = 40) -> dict:
    from playwright.async_api import async_playwright
    resultats, echecs_consecutifs = {}, 0
    async with async_playwright() as p:
        navigateur = await p.chromium.launch(headless=True, args=["--disable-blink-features=AutomationControlled", "--disable-dev-shm-usage"])
        contexte = await navigateur.new_context(locale="fr-FR", user_agent=UA)
        page = await contexte.new_page()
        await page.goto(accueil, wait_until="domcontentloaded", timeout=60000)
        await asyncio.sleep(3)
        for url in urls:
            donnees = None
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=60000)
                for _ in range(attente_max):
                    contenu = await page.evaluate(
                        "() => { const s = document.getElementById('__NEXT_DATA__'); return s ? s.textContent : null; }")
                    if contenu:
                        donnees = json.loads(contenu)
                        break
                    await asyncio.sleep(1)
            except Exception as e:
                print(f"  erreur de navigation ({type(e).__name__}) : nouvel onglet")
                page = await contexte.new_page()
            if donnees is None:
                echecs_consecutifs += 1
                print(f"  bloqué ou vide : {url}")
                if echecs_consecutifs >= 3:
                    print("  trois échecs de suite : arrêt de la collecte (voir IMPORTER_PAGES_HTML)")
                    break
                continue
            echecs_consecutifs = 0
            resultats[url] = donnees
            await asyncio.sleep(random.uniform(2.5, 5))
        await navigateur.close()
    return resultats


def recuperer(urls: list[str], accueil: str) -> dict:
    """Lance le navigateur dans un fil à part : fonctionne dans Colab comme en script."""
    with ThreadPoolExecutor(1) as ex:
        return ex.submit(asyncio.run, _recuperer(urls, accueil)).result()


def collecter_avis(url: str, max_pages: int) -> tuple[list[dict], dict]:
    hote, domaine, params = analyser_url(url)
    premiere = recuperer([url_page(hote, domaine, params, 1)], f"https://{hote}/")
    if not premiere:
        return [], {}
    props = next(iter(premiere.values()))["props"]["pageProps"]
    total = props["filters"]["pagination"]["totalPages"]
    print(f"{props['businessUnit']['displayName']} : {props['filters']['pagination']['totalCount']} avis, "
          f"{total} pages (limite : {max_pages})")
    suite = recuperer([url_page(hote, domaine, params, n) for n in range(2, min(total, max_pages) + 1)], f"https://{hote}/")
    pages = [props] + [d["props"]["pageProps"] for d in suite.values()]
    return [a for pp in pages for a in pp.get("reviews", [])], props["businessUnit"]


def collecter_profils(ids: list[str], hote: str) -> dict:
    resultats = recuperer([f"https://{hote}/users/{i}" for i in ids], f"https://{hote}/")
    return {u.rsplit("/", 1)[-1]: d["props"]["pageProps"] for u, d in resultats.items()}


def importer_html() -> tuple[list[dict], dict, dict]:
    """Lit des pages Trustpilot enregistrées depuis un navigateur (pages d'avis et pages profil)."""
    if EN_COLAB:
        from google.colab import files
        fichiers = {nom: contenu.decode("utf-8", "ignore") for nom, contenu in files.upload().items()}
    else:
        dossier = Path(os.environ.get("DOSSIER_HTML", "pages_html"))
        fichiers = {f.name: f.read_text("utf-8", "ignore") for f in dossier.glob("*.htm*")}
    avis, entreprise, profils = {}, {}, {}
    for nom, html in fichiers.items():
        donnees = lire_next_data(html)
        if not donnees:
            print(f"  {nom} : pas de données Trustpilot dans ce fichier")
            continue
        pp = donnees["props"]["pageProps"]
        if "reviews" in pp:
            entreprise = pp.get("businessUnit", entreprise)
            avis.update({a["id"]: a for a in pp["reviews"]})
        elif "consumerServiceReviews" in pp:
            profils[pp["consumer"]["id"]] = pp
    print(f"Import : {len(avis)} avis, {len(profils)} profils")
    return list(avis.values()), entreprise, profils


if not IMPORTER_PAGES_HTML and not URL_TRUSTPILOT.strip():
    raise SystemExit("Saisir l'URL de la page Trustpilot dans la cellule « Paramètres » "
                     "(par exemple https://fr.trustpilot.com/review/<domaine>).")
if IMPORTER_PAGES_HTML:
    avis_bruts, entreprise, profils_bruts = importer_html()
else:
    avis_bruts, entreprise = collecter_avis(URL_TRUSTPILOT, MAX_PAGES_AVIS)
    profils_bruts = {}
    if COLLECTER_PROFILS and avis_bruts:
        ids = list(dict.fromkeys(a["consumer"]["id"] for a in avis_bruts))[:MAX_PROFILS]
        print(f"Profils des auteurs : {len(ids)} à collecter (≈ {len(ids) * 5 // 60 + 1} min)")
        profils_bruts = collecter_profils(ids, analyser_url(URL_TRUSTPILOT)[0])
if not avis_bruts:
    raise SystemExit("Aucun avis récupéré : cocher IMPORTER_PAGES_HTML et téléverser les pages enregistrées.")
NOM_ENTREPRISE = entreprise.get("displayName", "")
print(f"{len(avis_bruts)} avis et {len(profils_bruts)} profils prêts pour l'analyse")

# %% Tableau des avis et des profils


def date_creation_compte(auteur_id: str) -> pd.Timestamp | None:
    """Les identifiants Trustpilot sont des ObjectId : les 4 premiers octets donnent la date de création du compte."""
    try:
        return pd.Timestamp(datetime.fromtimestamp(int(auteur_id[:8], 16), tz=timezone.utc))
    except ValueError:
        return None


def ligne_avis(a: dict) -> dict:
    verif = (a.get("labels") or {}).get("verification") or {}
    auteur, dates, reponse = a.get("consumer") or {}, a.get("dates") or {}, a.get("reply") or {}
    return {
        "id": a["id"], "note": a["rating"], "titre": a.get("title") or "", "texte": a.get("text") or "",
        "langue": a.get("language") or "", "date_experience": (dates.get("experiencedDate") or "")[:10],
        "date_publication": dates.get("publishedDate"), "source": a.get("source") or "",
        "verification": verif.get("verificationLevel") or "", "auteur_id": auteur.get("id", ""),
        "auteur_nom": auteur.get("displayName", ""), "auteur_nb_avis": auteur.get("numberOfReviews"),
        "auteur_pays": auteur.get("countryCode", ""), "auteur_photo": auteur.get("hasImage"),
        "reponse_entreprise": bool(reponse.get("message")),
    }


def resume_profil(pp: dict) -> dict:
    avis = pp.get("consumerServiceReviews") or []
    dates = sorted(pd.Timestamp(a["dates"]["publishedDate"]) for a in avis)
    jours = Counter(d.date() for d in dates)
    return {
        "auteur_id": pp["consumer"]["id"],
        "nb_avis_total": (pp.get("consumerStatistics") or {}).get("reviewsCount", pp["consumer"].get("numberOfReviews")),
        "nb_avis_lus": len(avis),
        "part_1_etoile": round(sum(a["rating"] == 1 for a in avis) / len(avis), 2) if avis else None,
        "max_avis_meme_jour": max(jours.values(), default=0),
        "entreprises_notees": " | ".join(f'{(a.get("businessUnit") or {}).get("displayName", "?")}:{a["rating"]}' for a in avis),
        "dates_avis": dates,
    }


avis = pd.DataFrame([ligne_avis(a) for a in avis_bruts]).drop_duplicates("id")
avis["date_publication"] = pd.to_datetime(avis.date_publication, utc=True)
avis = avis.sort_values("date_publication").reset_index(drop=True)
avis["negatif"] = avis.note <= 2
avis["texte_complet"] = np.where(
    avis.apply(lambda r: r.texte.startswith(r.titre.rstrip("…").rstrip(".")), axis=1),
    avis.texte, avis.titre + "\n" + avis.texte)
profils = pd.DataFrame([resume_profil(pp) for pp in profils_bruts.values()]) if profils_bruts else pd.DataFrame(
    columns=["auteur_id", "nb_avis_total", "nb_avis_lus", "part_1_etoile", "max_avis_meme_jour", "entreprises_notees", "dates_avis"])
avis = avis.merge(profils, on="auteur_id", how="left")
print(avis.groupby("note").size().rename("nb avis").to_frame().T.to_string())

# %% Indicateurs de temps et de compte


def silence_avant(row) -> float | None:
    dates = row.dates_avis if isinstance(row.dates_avis, list) else []
    precedentes = [d for d in dates if d < row.date_publication - pd.Timedelta(hours=1)]
    return (row.date_publication - max(precedentes)).days if precedentes else None


creation = avis.auteur_id.map(date_creation_compte)
avis["compte_cree"] = creation.dt.date
avis["age_compte_jours"] = (avis.date_publication - creation).dt.days
avis["heure_paris"] = avis.date_publication.dt.tz_convert(PARIS).dt.hour
avis["jour_semaine"] = avis.date_publication.dt.tz_convert(PARIS).dt.day_name()
avis["delai_experience_jours"] = (avis.date_publication.dt.tz_localize(None).dt.normalize()
                                  - pd.to_datetime(avis.date_experience, errors="coerce")).dt.days
avis["silence_avant_jours"] = avis.apply(silence_avant, axis=1)
nb_total = avis.nb_avis_total.fillna(avis.auteur_nb_avis)
cible_unique = avis.entreprises_notees.fillna("").apply(
    lambda s: bool(s) and all(e.rsplit(":", 1)[0] == NOM_ENTREPRISE for e in s.split(" | ")))
avis["compte_jetable"] = (avis.age_compte_jours <= 1) & ((nb_total <= 1) | cible_unique)


def rafales(df: pd.DataFrame, fenetre: int = 30) -> pd.Series:
    """p-valeur (corrigée de Bonferroni) de la fenêtre de 30 jours la plus chargée contenant chaque avis du même sens."""
    p_min = pd.Series(1.0, index=df.index)
    for sens, groupe in df.groupby("negatif"):
        t = groupe.date_publication
        for idx, debut in t.items():
            dans = (t >= debut) & (t < debut + pd.Timedelta(days=fenetre))
            k = int(dans.sum())
            # Rythme de référence : un an de part et d'autre, limité à la période réellement collectée.
            ref_debut = max(debut - pd.Timedelta(days=365), t.min())
            ref_fin = min(debut + pd.Timedelta(days=365), t.max())
            voisins = ((t >= ref_debut) & (t <= ref_fin)).sum()
            duree = max((ref_fin - ref_debut).days, fenetre)
            attendu = max(voisins * fenetre / duree, 1e-6)
            p = min(1.0, poisson.sf(k - 1, attendu) * len(groupe))
            p_min.loc[t[dans].index] = np.minimum(p_min.loc[t[dans].index], p)
    return p_min


avis["rafale_p"] = rafales(avis)

# %% Style d'écriture et orthographe
MOT = re.compile(r"[a-zA-ZÀ-ÿœŒ]+(?:-[a-zA-ZÀ-ÿœŒ]+)*")
EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")
DETAILS = {
    "prix": r"\d+(?:[.,]\d+)?\s?(?:€|euros?\b|eur\b|ht\b|ttc\b|\$|£)",
    "lieu_agence": r"\b(?:agence|magasin|boutique|store|branch)\s+(?:de|d'|du|of|in)?\s?[A-ZÉ]\w+",
    "reference_ou_mesure": r"\b[A-Z]{1,4}\d{2,}[A-Z0-9-]*\b|\b\d+(?:[.,]\d+)?\s?(?:A|mA|V|W|kW|mm|cm|m)\b",
    "duree_date": r"\b\d+\s?(?:jours?|semaines?|mois|heures?|h|days?|weeks?)\b|\b(?:lundi|mardi|mercredi|jeudi|vendredi|samedi|dimanche|hier|ce matin)\b",
}
FORMULES = {
    "a_fuir": r"\b(?:à|a) fuire?\b|\bfuyez\b|\bà éviter\b|\bavoid\b",
    "vol_arnaque": r"\bvol(?:eurs?|é|er)?\b|\barnaqu\w*|\bescroc\w*|\bscam\w*|\bthie(?:f|ves)\b",
    "zero_etoile": r"\b0 étoile|\bzéro étoile|mettre 0|zero stars?",
}
correcteurs = {"fr": SpellChecker(language="fr"), "en": SpellChecker(language="en")}
mots_entreprise = {m.lower() for m in MOT.findall(NOM_ENTREPRISE + " " + entreprise.get("identifyingName", ""))}


def compter(motif: str, texte: str, drapeaux: int = re.IGNORECASE) -> int:
    return len(re.findall(motif, texte, drapeaux))


def style(texte: str, langue: str) -> dict:
    mots = MOT.findall(texte)
    phrases = [p.strip() for p in re.split(r"[.!?…]+", texte) if p.strip()]
    debuts = [p[0] for p in phrases if p[0].isalpha()]
    correcteur = correcteurs.get(langue)
    candidats = [m.lower() for m in mots if len(m) > 2 and not m[0].isupper() and m.lower() not in mots_entreprise]
    inconnus = sorted(correcteur.unknown(candidats)) if correcteur else []
    return {
        "nb_mots": len(mots),
        "mots_par_phrase": round(len(mots) / max(len(phrases), 1), 1),
        "mots_inconnus": " ".join(inconnus),
        "taux_mots_inconnus": round(len(inconnus) / len(candidats), 3) if correcteur and candidats else None,
        "part_mots_majuscules": round(sum(len(m) > 2 and m.isupper() for m in mots) / max(len(mots), 1), 3),
        "nb_exclamations": texte.count("!"),
        "ponctuation_repetee": compter(r"!{2,}|\?{2,}|[!?]{3,}", texte),
        "points_suspension": compter(r"\.{3,}|…", texte),
        "espace_avant_ponct_double": compter(r"\s[!?;:]", texte),
        "ponct_double_collee": compter(r"\w[!?;:]", texte),
        "apostrophe_courbe": texte.count("’"),
        "apostrophe_droite": texte.count("'"),
        "espace_avant_virgule": compter(r"\s,", texte),
        "sans_espace_apres_ponct": compter(r"[,.;:!?][A-Za-zÀ-ÿ]", texte),
        "debut_phrase_minuscule": round(sum(c.islower() for c in debuts) / max(len(debuts), 1), 2),
        "emojis": len(EMOJI.findall(texte)),
        **{k: compter(m, texte, 0 if k == "reference_ou_mesure" else re.IGNORECASE) for k, m in DETAILS.items()},
        **{k: compter(m, texte) for k, m in FORMULES.items()},
    }


avis = pd.concat([avis, pd.DataFrame([style(t, l) for t, l in zip(avis.texte_complet, avis.langue)])], axis=1)
avis["nb_types_details"] = (avis[list(DETAILS)] > 0).sum(axis=1)


def languagetool(textes: pd.Series, langues: pd.Series, maximum: int) -> pd.DataFrame:
    """Erreurs détectées par l'API publique LanguageTool (20 requêtes/min), par catégorie."""
    lignes = {}
    for idx in textes.index[:maximum]:
        langue = {"fr": "fr", "en": "en-GB"}.get(langues[idx], "auto")
        try:
            r = requests.post("https://api.languagetool.org/v2/check",
                              data={"text": textes[idx][:5000], "language": langue}, timeout=30)
            r.raise_for_status()
            categories = Counter(m["rule"]["category"]["id"] for m in r.json()["matches"])
            lignes[idx] = {
                "lt_orthographe": categories.get("TYPOS", 0),
                "lt_grammaire": categories.get("GRAMMAR", 0) + categories.get("CONFUSED_WORDS", 0) + categories.get("AGREEMENT", 0),
                "lt_ponctuation_typo": categories.get("PUNCTUATION", 0) + categories.get("TYPOGRAPHY", 0) + categories.get("CASING", 0),
                "lt_total": sum(categories.values()),
            }
        except requests.RequestException as e:
            print(f"  LanguageTool indisponible ({e}) : arrêt de cette étape")
            break
        time.sleep(3.2)
    return pd.DataFrame.from_dict(lignes, orient="index")


if LANGUAGETOOL:
    ordre = avis.sort_values("negatif", ascending=False).index
    lt = languagetool(avis.texte_complet.loc[ordre], avis.langue, MAX_TEXTES_LANGUAGETOOL)
    avis = avis.join(lt)
    if "lt_total" in avis:
        avis["lt_erreurs_pour_100_mots"] = (100 * avis.lt_total / avis.nb_mots.clip(lower=1)).round(1)
print(avis.groupby("negatif")[["nb_mots", "taux_mots_inconnus", "nb_types_details", "nb_exclamations"]
                              + [c for c in ["lt_erreurs_pour_100_mots"] if c in avis]].median()
      .rename(index={False: "positifs/neutres", True: "négatifs"}).round(3).to_string())

# %% Noms des rédacteurs
NOMS_GENERIQUES = {"client", "consumer", "customer", "utilisateur", "user", "anonyme", "anonymous", "client déçu",
                   "cliente", "particulier", "pro", "un client", "une cliente", "mécontent", "déçu"}
FORME_SOCIETE = (r"\b(?:sarl|sas|sasu|eurl|sci|ets|établissements|services?|transports?|transfer|électricité|elec|"
                 r"bâtiment|batiment|conseil|groupe|société|company|ltd|limited|inc|gmbh|com|fr)\b|\.(?:com|fr)\b|\s-\s")


def signaux_nom(nom: str) -> dict:
    n = nom.strip()
    return {
        "nom_generique": n.lower() in NOMS_GENERIQUES or bool(re.fullmatch(r"(?:client|consumer|customer|user)\s?\w{0,3}", n.lower())),
        "nom_avec_chiffres": bool(re.search(r"\d", n)),
        "nom_prenom_initiale": bool(re.fullmatch(r"[A-ZÀ-Ý][a-zà-ÿ-]+\s[A-Z]\.?", n)),
        "nom_tout_majuscules": n.isupper() and len(n) > 3,
        "nom_forme_societe": bool(re.search(FORME_SOCIETE, n, re.I)),
    }


avis = pd.concat([avis, pd.DataFrame([signaux_nom(n) for n in avis.auteur_nom])], axis=1)
doublons_nom = avis.groupby("auteur_nom").auteur_id.nunique()
avis["nom_partage_par_plusieurs_comptes"] = avis.auteur_nom.map(doublons_nom) > 1
print(avis[["nom_generique", "nom_avec_chiffres", "nom_forme_societe", "nom_partage_par_plusieurs_comptes"]].sum().to_string())

# %% Jugements Jev (TypeSafe)


def cle_typesafe() -> str | None:
    if EN_COLAB:
        try:
            from google.colab import userdata
            return userdata.get("TYPESAFE_API_KEY")
        except Exception:
            return None
    return os.environ.get("TYPESAFE_API_KEY")


QUESTIONS_JEV = {
    "experience_precise": {"type": "noul", "instructions":
        "L'avis `avis.texte` raconte-t-il une expérience vécue précise, avec au moins un élément concret et situé "
        "(un lieu ou une agence, un produit, un montant, un échange avec le personnel, une chronologie) ?"},
    "plainte_generique": {"type": "noul", "instructions":
        "L'avis `avis.texte` est-il une appréciation générique et interchangeable, sans aucun fait particulier, "
        "qui pourrait viser n'importe quelle entreprise du même secteur ?"},
    "style_ia": {"type": "noul", "instructions":
        "Ce texte `avis.texte` ressemble-t-il à un texte généré par une intelligence artificielle (formules toutes "
        "faites, structure trop régulière, emphase, vocabulaire lisse, aucune faute ni oralité) ?"},
    "invraisemblance": {"type": "noul", "instructions":
        "Le récit de l'avis `avis.texte` sur l'entreprise `avis.entreprise` contient-il une contradiction interne "
        "ou un élément invraisemblable ?"},
    "promotion_concurrent": {"type": "noul", "instructions":
        "L'avis `avis.texte` cherche-t-il surtout à orienter le lecteur vers un concurrent nommé, plutôt qu'à "
        "raconter sa propre expérience ?"},
    "nom_entreprise": {"type": "noul", "instructions":
        "Le nom affiché `avis.auteur` est-il un nom d'entreprise, de commerce ou d'organisation plutôt qu'un nom "
        "de personne ou un pseudonyme ?"},
    "sentiment": {"type": "score", "instructions":
        "Quel est le sentiment exprimé par le texte `avis.texte` envers l'entreprise, indépendamment de la note donnée ?",
        "criteria": ["Très négatif : colère, accusation, déconseille fermement.",
                     "Plutôt négatif : déception, reproches dominants.",
                     "Mitigé : points positifs et négatifs équilibrés.",
                     "Plutôt positif : satisfaction avec réserves mineures.",
                     "Très positif : enthousiasme, recommande sans réserve."]},
    "qualite_informative": {"type": "score", "instructions":
        "À quel point l'avis `avis.texte` est-il utile à un futur client pour se faire une idée de l'entreprise ?",
        "criteria": ["Inutile : une émotion ou un mot, aucun fait.",
                     "Faible : un reproche ou un compliment vague.",
                     "Utile : au moins un fait précis et compréhensible.",
                     "Très utile : récit circonstancié, vérifiable, nuancé."]},
    "profil_auteur": {"type": "choice", "instructions": "D'après le texte `avis.texte`, qui est l'auteur de l'avis ?",
        "criteria": {"professionnel": "Achète pour son activité professionnelle (artisan, entreprise, revendeur).",
                     "particulier": "Achète pour un usage personnel ou son logement.",
                     "indeterminable": "Le texte ne permet pas de le dire."}},
    **{f"theme_{k}": {"type": "noul", "instructions": f"L'avis `avis.texte` parle-t-il de {v} ?"} for k, v in THEMES.items()},
}

# Témoins à réponse connue : si l'un échoue, les verdicts de la question concernée sont à prendre avec réserve.
TEMOINS = [
    ("precis", "Passé au magasin de Brest mardi pour une cartouche de filtre réf. FX-220, 14,90 €. Le vendeur m'a dit de "
     "revenir jeudi, rupture. Jeudi toujours rien, et personne ne m'a rappelé.", 1, "Marc L",
     {"experience_precise": True, "plainte_generique": False, "nom_entreprise": False, "sentiment": 0}),
    ("generique", "Nul. Service horrible, à fuir. Je déconseille.", 1, "SARL Martin et Fils",
     {"experience_precise": False, "plainte_generique": True, "nom_entreprise": True, "sentiment": 0}),
    ("ia_negatif", "Je tiens à partager mon expérience profondément décevante avec cette entreprise. Dès le premier "
     "contact, j'ai été frappé par le manque de professionnalisme. En somme, il est essentiel de rappeler que la "
     "satisfaction client devrait être au cœur des priorités. Je ne peux que vous recommander de vous tourner vers "
     "d'autres solutions.", 1, "Sophie", {"style_ia": True, "sentiment": 0}),
    ("ia_positif", "Une expérience exceptionnelle du début à la fin ! L'équipe a su faire preuve d'une écoute "
     "remarquable et d'un professionnalisme irréprochable. Que vous soyez un particulier ou un professionnel, "
     "n'hésitez pas : vous ne serez pas déçu. Je recommande vivement !", 5, "Julie D", {"style_ia": True, "sentiment": 4}),
    ("oral", "bon ben jai attendu 3 semaines ma commande et toujours rien.. le gars au comptoir s'en fout "
     "royalement, jirai chez le concurent la prochaine fois", 1, "kevin33", {"style_ia": False, "experience_precise": True}),
    ("concurrent", "Trop cher. Allez chez PrixMalin Discount, livraison gratuite et prix 30 % moins chers, le meilleur "
     "site du marché, je recommande PrixMalin à tous !", 1, "Paul", {"promotion_concurrent": True}),
    ("invraisemblable", "J'ai commandé un canapé à 9 h, livré à 9 h 15 à 600 km de l'entrepôt, "
     "mais il manquait tout car ils n'avaient jamais vendu de meubles.", 1, "Luc", {"invraisemblance": True}),
]


def appeler_jev(texte: str, note: int, auteur: str, cle: str) -> dict:
    corps = {"model": "jev-latest", "questions": QUESTIONS_JEV,
             "state": {"avis": {"texte": texte, "note_sur_5": int(note), "auteur": auteur, "entreprise": NOM_ENTREPRISE}}}
    for essai in range(3):
        try:
            r = requests.post("https://api.typesafe.ai/v1/systemone", json=corps, timeout=120,
                              headers={"Authorization": f"Bearer {cle}"})
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if essai == 2:
                raise
            time.sleep(2 * (essai + 1))


def aplatir_jev(reponse: dict) -> dict:
    r = reponse["answers"]
    sortie = {f"jev_{k}": round(r[k]["noul"], 3) for k, q in QUESTIONS_JEV.items() if q["type"] == "noul"}
    sortie["jev_sentiment_0_4"] = round(r["sentiment"]["score"], 2)
    sortie["jev_qualite_0_3"] = round(r["qualite_informative"]["score"], 2)
    sortie["jev_profil_auteur"] = r["profil_auteur"].get("choice")
    sortie["jev_jetons"] = reponse.get("usage", {}).get("input_tokens", 0)
    return sortie


def echecs_temoin(res: dict, attendus: dict) -> list[str]:
    echecs = []
    for q, attendu in attendus.items():
        if q == "sentiment":
            if abs(res["jev_sentiment_0_4"] - attendu) > 1:
                echecs.append(q)
        elif (res[f"jev_{q}"] >= 0.5) != attendu:
            echecs.append(q)
    return echecs


cle = cle_typesafe() if JUGEMENTS_JEV else None
questions_douteuses = set()
if cle:
    with ThreadPoolExecutor(8) as pool:
        rep_temoins = list(pool.map(lambda t: appeler_jev(t[1], t[2], t[3], cle), TEMOINS))
        rep_avis = list(pool.map(lambda r: appeler_jev(r[0], r[1], r[2], cle),
                                 zip(avis.texte_complet, avis.note, avis.auteur_nom)))
    for (nom, *_, attendus), rep in zip(TEMOINS, rep_temoins):
        echecs = echecs_temoin(aplatir_jev(rep), attendus)
        questions_douteuses.update(echecs)
        print(f"  témoin {nom:16s} {'OK' if not echecs else 'ÉCHEC : ' + ', '.join(echecs)}")
    jev = pd.DataFrame([aplatir_jev(r) for r in rep_avis], index=avis.index)
    avis = avis.join(jev)
    jetons = int(jev.jev_jetons.sum())
    print(f"Jev : {len(jev)} avis jugés, {jetons} jetons (≈ {jetons * 0.042 / 1e6:.3f} $)."
          + (f" Questions à prendre avec réserve : {', '.join(sorted(questions_douteuses))}" if questions_douteuses else " Tous les témoins passent."))
else:
    print("Jev non utilisé (case décochée ou clé TYPESAFE_API_KEY absente).")

# %% Sentiment par modèle local (comparé à la note)
# Avis en français : modèle entraîné sur des avis français ; autres langues : modèle multilingue.
# La note lue est l'espérance des étoiles (somme des probabilités pondérées), plus stable que l'étoile la plus probable.
MODELE_SENTIMENT_FR = "cmarkea/distilcamembert-base-sentiment"
MODELE_SENTIMENT_MULTILINGUE = "nlptown/bert-base-multilingual-uncased-sentiment"


def etoiles_attendues(textes: list[str], modele: str) -> list[float]:
    if not textes:
        return []
    import torch
    from transformers import pipeline
    # Explicite : les versions 4.x de transformers restent sur le processeur si rien n'est précisé.
    classifieur = pipeline("sentiment-analysis", model=modele, truncation=True, max_length=512, top_k=None,
                           device=0 if torch.cuda.is_available() else -1)
    return [round(sum(int(d["label"][0]) * d["score"] for d in s), 2) for s in classifieur(textes, batch_size=16)]


if SENTIMENT_LOCAL:
    en_francais = avis.langue.eq("fr")
    avis["sentiment_modele_etoiles"] = np.nan
    for masque, modele in ((en_francais, MODELE_SENTIMENT_FR), (~en_francais, MODELE_SENTIMENT_MULTILINGUE)):
        avis.loc[masque, "sentiment_modele_etoiles"] = etoiles_attendues(avis.texte_complet[masque].tolist(), modele)
    avis["ecart_note_sentiment_modele"] = (avis.sentiment_modele_etoiles - avis.note).abs()
    print(pd.crosstab(avis.note, avis.sentiment_modele_etoiles.round().astype(int),
                      rownames=["note donnée"], colnames=["note lue dans le texte (arrondie)"]))

# %% Ressemblance entre textes (copie, paraphrase, même plume)
MOTS_OUTILS = sorted(set((
    "le la les un une des de du au aux et ou mais donc car ni que qui quoi dont où ce cet cette ces je j me m moi "
    "tu il elle on nous vous ils elles se s mon ma mes son sa ses leur leurs ne n pas plus jamais rien très trop bien "
    "tout tous toute aussi même encore déjà alors puis en dans sur sous pour par avec sans chez vers avant après "
    "the a an and or but of to in on for with at by from is are was were it this that i my me we you they not very").split()))
HABITUDES = ["nb_exclamations", "ponctuation_repetee", "points_suspension", "espace_avant_ponct_double",
             "ponct_double_collee", "apostrophe_courbe", "apostrophe_droite", "espace_avant_virgule",
             "sans_espace_apres_ponct", "debut_phrase_minuscule", "part_mots_majuscules", "mots_par_phrase"]

textes = avis.texte_complet.str.lower().tolist()
sim_mots = cosine_similarity(TfidfVectorizer(ngram_range=(1, 2)).fit_transform(textes))
sim_car = cosine_similarity(TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit_transform(textes))
sim_outils = cosine_similarity(TfidfVectorizer(vocabulary=MOTS_OUTILS, token_pattern=r"(?u)\b\w+\b").fit_transform(textes))
sim_sens = None
if SIMILARITE_SEMANTIQUE:
    from sentence_transformers import SentenceTransformer
    modele_sens = SentenceTransformer("BAAI/bge-m3")
    sim_sens = cosine_similarity(modele_sens.encode(avis.texte_complet.tolist(), batch_size=16, normalize_embeddings=True))
h = avis[HABITUDES].fillna(0).to_numpy(dtype=float)
h = (h - h.mean(0)) / (h.std(0) + 1e-9)

lignes = []
for i in range(len(avis)):
    for j in range(i + 1, len(avis)):
        if avis.auteur_id[i] == avis.auteur_id[j]:
            continue
        lignes.append({
            "i": i, "j": j, "auteur_a": avis.auteur_nom[i], "note_a": avis.note[i],
            "auteur_b": avis.auteur_nom[j], "note_b": avis.note[j],
            "ecart_jours": abs((avis.date_publication[i] - avis.date_publication[j]).days),
            "sim_mots": round(sim_mots[i, j], 3), "sim_caracteres": round(sim_car[i, j], 3),
            "sim_sens": round(sim_sens[i, j], 3) if sim_sens is not None else None,
            "sim_mots_outils": round(sim_outils[i, j], 3),
            "distance_habitudes": round(float(np.linalg.norm(h[i] - h[j])), 2),
        })
paires = pd.DataFrame(lignes)
tri = "sim_sens" if sim_sens is not None else "sim_caracteres"
paires = paires.sort_values(tri, ascending=False).reset_index(drop=True) if len(paires) else paires


def max_par_avis(colonne: str) -> pd.Series:
    if not len(paires) or paires[colonne].isna().all():
        return pd.Series(np.nan, index=avis.index)
    return pd.concat([paires.groupby("i")[colonne].max(), paires.groupby("j")[colonne].max()], axis=1).max(axis=1).reindex(avis.index)


avis["sim_max_caracteres"] = max_par_avis("sim_caracteres")
avis["sim_max_sens"] = max_par_avis("sim_sens")
print("Paires d'auteurs différents les plus proches :")
if len(paires):
    display(paires.head(10).merge(avis[["texte_complet"]], left_on="i", right_index=True)
            .merge(avis[["texte_complet"]], left_on="j", right_index=True, suffixes=("_a", "_b")))

# %% Vocabulaire et expressions récurrentes
VIDES = set(MOTS_OUTILS) | {"est", "suis", "été", "ai", "avez", "sont", "pas", "plus", "it", "is"}


def mots_distinctifs(groupe_a: pd.Series, groupe_b: pd.Series, n: int = 15) -> pd.DataFrame:
    """Log-odds informatif avec a priori de Dirichlet (Monroe et al., 2008) : mots propres au groupe A."""
    ca = Counter(m for t in groupe_a for m in MOT.findall(t.lower()) if m not in VIDES and len(m) > 2)
    cb = Counter(m for t in groupe_b for m in MOT.findall(t.lower()) if m not in VIDES and len(m) > 2)
    prior = ca + cb
    a0, na, nb = sum(prior.values()), sum(ca.values()), sum(cb.values())
    lignes = []
    for mot, alpha in prior.items():
        ya, yb = ca[mot], cb[mot]
        delta = (math.log((ya + alpha) / (na + a0 - ya - alpha)) - math.log((yb + alpha) / (nb + a0 - yb - alpha)))
        var = 1 / (ya + alpha) + 1 / (yb + alpha)
        lignes.append({"mot": mot, "z": round(delta / math.sqrt(var), 2), "occurrences": ya})
    return pd.DataFrame(lignes).sort_values("z", ascending=False).head(n)


def expressions_partagees(df: pd.DataFrame, n: int = 3, min_auteurs: int = 3) -> pd.DataFrame:
    auteurs = defaultdict(set)
    for texte, auteur in zip(df.texte_complet, df.auteur_id):
        mots = MOT.findall(texte.lower())
        for k in range(len(mots) - n + 1):
            gramme = tuple(mots[k:k + n])
            if sum(m not in VIDES for m in gramme) >= 2:
                auteurs[" ".join(gramme)].add(auteur)
    lignes = [{"expression": e, "nb_auteurs": len(a)} for e, a in auteurs.items() if len(a) >= min_auteurs]
    return pd.DataFrame(lignes, columns=["expression", "nb_auteurs"]).sort_values("nb_auteurs", ascending=False)


if avis.negatif.any() and (~avis.negatif).any():
    display(Markdown("**Mots propres aux avis négatifs**"))
    display(mots_distinctifs(avis[avis.negatif].texte_complet, avis[~avis.negatif].texte_complet).T)
expr = expressions_partagees(avis)
display(Markdown("**Expressions de 3 mots reprises par au moins 3 auteurs différents**"))
display(expr.head(20) if len(expr) else "aucune")

# %% Registre des entreprises (France) pour les noms de société


def chercher_registre(nom: str) -> dict:
    requete = re.split(r"\s-\s|,", nom)[0].strip()
    try:
        r = requests.get("https://recherche-entreprises.api.gouv.fr/search",
                         params={"q": requete, "per_page": 5}, timeout=30)
        r.raise_for_status()
        resultats = r.json().get("results", [])
    except requests.RequestException as e:
        return {"registre_statut": f"erreur : {e}"}
    norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
    if not norm(requete):
        return {"registre_statut": "nom non comparable"}
    exacts = [x for x in resultats if norm(requete) in {norm(x.get("nom_complet", "")), norm(x.get("nom_raison_sociale") or ""), norm(x.get("sigle") or "")}]
    if not exacts:
        return {"registre_statut": "aucune correspondance exacte"}
    x = exacts[0]
    fiche = {"registre_statut": "fermée" if x.get("etat_administratif") == "C" else "active",
             "registre_nom": x.get("nom_complet"), "registre_siren": x.get("siren"),
             "registre_creation": x.get("date_creation"), "registre_fermeture": x.get("date_fermeture"),
             "registre_activite": x.get("activite_principale"), "registre_homonymes": len(exacts) - 1}
    try:
        b = requests.get("https://bodacc-datadila.opendatasoft.com/api/explore/v2.1/catalog/datasets/annonces-commerciales/records",
                         params={"where": f'registre like "{x["siren"]}"', "limit": 100, "order_by": "dateparution desc"},
                         timeout=30)
        b.raise_for_status()
        annonces = b.json().get("results", [])
        lire = lambda a, champ: json.loads(a[champ]) if isinstance(a.get(champ), str) else {}
        comptes = [pd.to_datetime(lire(a, "depot").get("dateCloture"), errors="coerce", dayfirst="/" in str(lire(a, "depot").get("dateCloture")))
                   for a in annonces if a.get("depot")]
        comptes = [c for c in comptes if pd.notna(c)]
        jugements = [lire(a, "jugement") for a in annonces if a.get("jugement")]
        fiche["registre_derniers_comptes"] = max(comptes).date().isoformat() if comptes else None
        fiche["registre_procedures"] = " ; ".join(f'{j.get("nature") or "?"} ({j.get("date") or "?"})' for j in jugements)
    except (requests.RequestException, ValueError, KeyError, TypeError) as e:
        fiche["registre_procedures"] = f"BODACC indisponible : {type(e).__name__}"
    return fiche


registre = pd.DataFrame()
if VERIFIER_REGISTRE:
    proba_societe = avis.get("jev_nom_entreprise", pd.Series(0, index=avis.index))
    candidats = avis[(proba_societe >= 0.6) | avis.nom_forme_societe].drop_duplicates("auteur_id")
    fiches = {aid: chercher_registre(nom) for aid, nom in zip(candidats.auteur_id, candidats.auteur_nom)}
    registre = pd.DataFrame.from_dict(fiches, orient="index")
    if len(registre):
        avis = avis.merge(registre, left_on="auteur_id", right_index=True, how="left")
        cols = ["auteur_nom", "date_publication", "note"] + list(registre.columns)
        display(avis[avis.auteur_id.isin(registre.index)][cols])
    print(f"{len(candidats)} noms de société vérifiés au registre (une correspondance de nom n'est pas une preuve d'identité)")

# %% Indice de vigilance


def col(nom: str, defaut=np.nan) -> pd.Series:
    return avis[nom] if nom in avis else pd.Series(defaut, index=avis.index)


seuil = SEUILS["jev"]
ecart_sentiment = (col("jev_sentiment_0_4") - (avis.note - 1)).abs()
ecart_modele = col("ecart_note_sentiment_modele")
fermeture = pd.to_datetime(col("registre_fermeture"), errors="coerce").dt.tz_localize("UTC")
SIGNAUX = {
    "texte_recopie": avis.sim_max_caracteres >= SEUILS["recopie_caracteres"],
    "paraphrase_autre_auteur": col("sim_max_sens") >= SEUILS["paraphrase_semantique"],
    "style_ia": (col("jev_style_ia") >= seuil) & ("style_ia" not in questions_douteuses),
    # Le modèle local lit mal les constats neutres : il ne déclenche le signal que si Jev est absent ou va dans son sens.
    "incoherence_note_texte": (ecart_sentiment >= 2) | ((ecart_modele >= SEUILS["ecart_sentiment_modele"]) & ~(ecart_sentiment < 1)),
    "promotion_concurrent": col("jev_promotion_concurrent") >= seuil,
    "generique_sans_fait": (col("jev_plainte_generique") >= seuil) & (col("jev_experience_precise") < 0.3),
    "invraisemblance": col("jev_invraisemblance") >= seuil,
    "compte_jetable": avis.compte_jetable,
    "dans_une_rafale": avis.rafale_p <= SEUILS["rafale_p"],
    "serie_meme_jour": col("max_avis_meme_jour", 0) >= SEUILS["meme_jour"],
    "societe_fermee": (col("registre_statut", "") == "fermée") & (fermeture < avis.date_publication),
    "reveil_apres_long_silence": col("silence_avant_jours") >= SEUILS["silence_jours"],
    "nom_generique": avis.nom_generique,
}
signaux = pd.DataFrame({k: v.fillna(False).astype(bool) for k, v in SIGNAUX.items()})
avis["indice_vigilance"] = (signaux.mul(pd.Series(POIDS)).sum(axis=1) / sum(POIDS.values()) * 10).round(1)
avis["raisons"] = signaux.apply(lambda r: ", ".join(k for k, v in r.items() if v), axis=1)

synthese = signaux.groupby(avis.negatif).mean().T.rename(columns={False: "positifs/neutres", True: "négatifs"})
display(Markdown("**Part des avis concernés par chaque signal**"))
display((synthese * 100).round(0).astype(int).astype(str) + " %")
colonnes_vue = ["date_publication", "note", "auteur_nom", "indice_vigilance", "raisons", "texte_complet"]
display(Markdown("**Avis à relire en priorité**"))
display(avis.sort_values("indice_vigilance", ascending=False)[colonnes_vue].head(15))

# %% Graphiques
avis["mois"] = avis.date_publication.dt.tz_convert(PARIS).dt.tz_localize(None).dt.to_period("M").dt.to_timestamp()
couleurs = {1: "#c0392b", 2: "#e67e22", 3: "#f1c40f", 4: "#7fb04f", 5: "#2e8b57"}
fig = px.histogram(avis, x="mois", color="note", color_discrete_map=couleurs, category_orders={"note": [1, 2, 3, 4, 5]},
                   title="Avis publiés par mois, selon la note")
fig.update_layout(bargap=0.1, xaxis_title="", yaxis_title="avis")
fig.show()

fig = px.scatter(avis, x="date_publication", y=avis.age_compte_jours.clip(lower=0) + 1, color=avis.note.astype(str),
                 color_discrete_map={str(k): v for k, v in couleurs.items()}, log_y=True, hover_name="auteur_nom",
                 hover_data=["nb_avis_total", "raisons"], title="Âge du compte au moment de l'avis (jours, échelle log)")
fig.update_layout(yaxis_title="âge du compte + 1 (jours)", xaxis_title="", legend_title="note")
fig.show()

fig = px.histogram(avis, x="heure_paris", color=avis.negatif.map({True: "négatifs", False: "positifs/neutres"}),
                   barmode="group", nbins=24, title="Heure de publication (heure de Paris)")
fig.update_layout(xaxis_title="heure", yaxis_title="avis", legend_title="")
fig.show()

themes_cols = [c for c in avis if c.startswith("jev_theme_")]
if themes_cols and avis.negatif.any():
    neg = avis[avis.negatif]
    parts = (neg[themes_cols] >= 0.5).groupby(neg.date_publication.dt.year).mean().mul(100).round(0)
    parts.columns = [c.replace("jev_theme_", "") for c in parts.columns]
    fig = px.imshow(parts.T, text_auto=True, color_continuous_scale="Reds", aspect="auto",
                    title="Thèmes des avis négatifs par année (% des avis)")
    fig.update_layout(xaxis_title="", yaxis_title="")
    fig.show()

fig = px.histogram(avis, x="indice_vigilance", color=avis.negatif.map({True: "négatifs", False: "positifs/neutres"}),
                   barmode="overlay", nbins=20, title="Indice de vigilance (0 à 10)")
fig.update_layout(xaxis_title="indice", yaxis_title="avis", legend_title="")
fig.show()

# %% Export (Excel, et Google Drive si coché)
domaine = entreprise.get("identifyingName") or (analyser_url(URL_TRUSTPILOT)[1] if URL_TRUSTPILOT.strip() else "page")
horodatage = datetime.now().strftime("%Y%m%d_%H%M")
nom_fichier = f"authenticite_{domaine}_{horodatage}.xlsx"
if ENREGISTRER_SUR_DRIVE and EN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    dossier = Path("/content/drive/MyDrive") / DOSSIER_DRIVE
else:
    dossier = Path("sorties")
dossier.mkdir(parents=True, exist_ok=True)
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE


def sans_tz(df: pd.DataFrame) -> pd.DataFrame:
    """Excel refuse les dates avec fuseau et certains caractères de contrôle."""
    dates = {c: df[c].dt.tz_convert(PARIS).dt.tz_localize(None) for c in df.select_dtypes(include=["datetimetz"]).columns}
    textes = {c: df[c].map(lambda v: ILLEGAL_CHARACTERS_RE.sub("", v) if isinstance(v, str) else v)
              for c in df.select_dtypes(include=["object"]).columns}
    return df.assign(**dates, **textes)
with pd.ExcelWriter(dossier / nom_fichier) as xl:
    sans_tz(avis.drop(columns=["dates_avis"], errors="ignore")).to_excel(xl, sheet_name="avis", index=False)
    signaux.assign(id=avis.id).to_excel(xl, sheet_name="signaux", index=False)
    sans_tz(paires.head(500)).to_excel(xl, sheet_name="paires_proches", index=False)
    if len(expr):
        expr.to_excel(xl, sheet_name="expressions", index=False)
    pd.Series(POIDS, name="poids").to_excel(xl, sheet_name="poids")
print(f"Résultats : {dossier / nom_fichier}")
if EN_COLAB and not ENREGISTRER_SUR_DRIVE:
    from google.colab import files
    files.download(str(dossier / nom_fichier))

# %% Conclusion
LIBELLES = {
    "texte_recopie": "texte recopié d'un autre auteur", "paraphrase_autre_auteur": "paraphrase d'un autre auteur",
    "style_ia": "style d'IA", "incoherence_note_texte": "note contredite par le texte",
    "promotion_concurrent": "promotion d'un concurrent", "generique_sans_fait": "texte générique sans fait",
    "invraisemblance": "récit invraisemblable", "compte_jetable": "compte créé pour l'avis",
    "dans_une_rafale": "publié dans une rafale", "serie_meme_jour": "auteur en série le même jour",
    "societe_fermee": "société signataire fermée", "reveil_apres_long_silence": "réveil après un long silence",
    "nom_generique": "nom générique",
}
CATEGORIES = {"sans signal d'alerte": "#2e8b57", "signal isolé": "#f1c40f", "à relire": "#c0392b"}


def pct(x: float) -> str:
    return f"{x * 100:.0f} %"


def majuscule(texte: str) -> str:
    return texte[:1].upper() + texte[1:]


def virgule(x: float) -> str:
    return f"{x:.1f}".replace(".", ",")


categorie = pd.Series(np.select([avis.indice_vigilance >= SEUILS["relecture"], avis.indice_vigilance > 0],
                                ["à relire", "signal isolé"], "sans signal d'alerte"), index=avis.index)
groupe = avis.negatif.map({True: "négatifs", False: "positifs ou neutres"})
a_relire = categorie.eq("à relire")
part_sans_signal = categorie.eq("sans signal d'alerte").mean()
parts_groupe = {g: a_relire[groupe == g].mean() for g in groupe.unique()}

# Graphique : part globale (anneau) et détail par type d'avis (barres empilées).
from plotly.subplots import make_subplots
global_ = categorie.value_counts().reindex(list(CATEGORIES), fill_value=0)
detail = pd.crosstab(groupe, categorie, normalize="index").reindex(columns=list(CATEGORIES), fill_value=0) * 100
fig = make_subplots(rows=1, cols=2, column_widths=[0.4, 0.6], specs=[[{"type": "domain"}, {"type": "xy"}]],
                    subplot_titles=("Ensemble des avis", "Détail par type d'avis"))
fig.add_trace(go.Pie(labels=global_.index, values=global_.values, hole=0.62, sort=False, direction="clockwise",
                     marker_colors=list(CATEGORIES.values()), textinfo="percent", showlegend=True), 1, 1)
for nom, couleur in CATEGORIES.items():
    fig.add_trace(go.Bar(y=detail.index, x=detail[nom], name=nom, orientation="h", marker_color=couleur,
                         text=detail[nom].round(0).astype(int).astype(str) + " %", textposition="inside",
                         showlegend=False), 1, 2)
fig.update_layout(barmode="stack", title="Part des avis sans signal d'alerte", height=380,
                  annotations=list(fig.layout.annotations) + [dict(
                      text=f"<b>{pct(part_sans_signal)}</b><br>sans signal",
                      x=0.2, y=0.5, xref="paper", yref="paper", showarrow=False, font_size=18)])
fig.update_xaxes(range=[0, 100], ticksuffix=" %", row=1, col=2)
fig.show()

# Constats chiffrés.
lignes = []
debut, fin = avis.date_publication.min().tz_convert(PARIS), avis.date_publication.max().tz_convert(PARIS)
total_page = entreprise.get("numberOfReviews")
lignes.append(f"**Corpus.** {len(avis)} avis de {NOM_ENTREPRISE or 'cette entreprise'} publiés du "
              f"{debut:%d/%m/%Y} au {fin:%d/%m/%Y}, dont {int(avis.negatif.sum())} négatifs (1 ou 2 étoiles) ; "
              f"note moyenne {virgule(avis.note.mean())}/5. {len(profils_bruts)} profils d'auteurs lus."
              + (f" Trustpilot en annonce {total_page} : l'écart vient de la limite de pages ou d'avis non affichés."
                 if total_page and total_page > len(avis) else ""))
lignes.append(f"**Résultat global.** {pct(part_sans_signal)} des avis ne présentent aucun "
              f"signal d'alerte, {pct(global_['signal isolé'] / len(avis))} un signal isolé (souvent anodin) et "
              f"{pct(a_relire.mean())} assez de signaux pour être relus ({int(a_relire.sum())} avis, indice ≥ "
              f"{virgule(SEUILS['relecture'])}). Par type : "
              + ", ".join(f"{pct(p)} des avis {g}" for g, p in parts_groupe.items()) + " sont à relire.")

ecarts = []
if avis.negatif.any() and (~avis.negatif).any():
    for s in signaux:
        pn, pp = signaux[s][avis.negatif].mean(), signaux[s][~avis.negatif].mean()
        if max(pn, pp) >= 0.10 and abs(pn - pp) >= 0.10:
            ecarts.append((abs(pn - pp), f"{LIBELLES[s]} ({pct(pn)} des négatifs contre {pct(pp)} des positifs ou neutres)"))
lignes.append("**Ce qui distingue les deux camps.** " + (majuscule(" ; ".join(t for _, t in sorted(ecarts, reverse=True))) + "."
              if ecarts else "Aucun signal ne sépare nettement les avis négatifs des autres."))

nb_rafale, nb_serie = int(signaux.dans_une_rafale.sum()), int(signaux.serie_meme_jour.sum())
nb_copies = int((paires.sim_caracteres >= SEUILS["recopie_caracteres"]).sum()) if len(paires) else 0
nb_paraphrases = int((paires.sim_sens >= SEUILS["paraphrase_semantique"]).sum()) if len(paires) and sim_sens is not None else 0
mois_rafale = ""
if nb_rafale:
    rafale = avis[signaux.dans_une_rafale]
    mois = rafale.date_publication.dt.tz_convert(PARIS).dt.strftime("%m/%Y")
    mois_rafale = " (" + ", ".join(f"{m} : {n} {g}" for (m, g), n in rafale.groupby([mois, groupe[rafale.index]])
                                   .size().sort_values(ascending=False).head(4).items()) + ")"
coordination = [t for n, t in [(nb_copies, f"{nb_copies} paire(s) de textes recopiés entre auteurs différents"),
                               (nb_paraphrases, f"{nb_paraphrases} paire(s) de paraphrases"),
                               (nb_rafale, f"{nb_rafale} avis publiés dans des rafales{mois_rafale}"),
                               (nb_serie, f"{nb_serie} avis d'auteurs qui en publient en série le même jour")] if n]
lignes.append("**Indices de coordination.** " + (majuscule(" ; ".join(coordination)) + "." if coordination else
              "Aucun : ni rafale anormale, ni texte recopié, ni auteur en série."))

if a_relire.any():
    tete = avis[a_relire].sort_values("indice_vigilance", ascending=False).head(3)
    lignes.append("**Avis les plus marqués.** " + " ; ".join(
        f"{r.date_publication.tz_convert(PARIS):%d/%m/%Y}, note {r.note}/5, indice {virgule(r.indice_vigilance)} : "
        + ", ".join(LIBELLES[k] for k in r.raisons.split(", ") if k) for r in tete.itertuples()) + ".")

reserves = []
if not profils_bruts:
    reserves.append("profils des auteurs non lus (séries le même jour et réveils après un long silence non évalués)")
if not cle:
    reserves.append("Jev non utilisé (style d'IA, texte générique, invraisemblance et promotion d'un concurrent non évalués)")
if questions_douteuses:
    reserves.append(f"témoins Jev en échec sur {', '.join(sorted(questions_douteuses))}")
if len(avis) < 50:
    reserves.append("moins de 50 avis : les tests de rafale ne repèrent que des écarts grossiers")
if reserves:
    lignes.append("**Réserves.** " + majuscule(" ; ".join(reserves)) + ".")

# Verdict : prudent par construction, il décrit un schéma sans conclure à la fraude.
ecart_camps = max(parts_groupe.values()) - min(parts_groupe.values()) if len(parts_groupe) == 2 else 0
camp = max(parts_groupe, key=parts_groupe.get)
if a_relire.mean() < 0.05 and not coordination:
    verdict = ("Rien n'indique une manipulation organisée : les signaux relevés restent isolés, presque aucun avis "
               "n'en cumule plusieurs et aucun indice de coordination n'apparaît. Les avis paraissent dans leur "
               "ensemble spontanés.")
elif ecart_camps >= 0.15 and coordination:
    verdict = (f"Les signaux se concentrent sur les avis {camp} et s'accompagnent d'indices de coordination. Ce "
               "schéma est compatible avec une campagne ciblée, sans la prouver : relire les avis listés plus haut "
               "avant toute conclusion.")
elif ecart_camps >= 0.15:
    verdict = (f"Les avis {camp} portent nettement plus de signaux que les autres, mais sans indice de coordination : "
               "cela évoque des comptes isolés plutôt qu'une campagne.")
else:
    verdict = ("Des signaux existent mais se répartissent de façon comparable entre avis positifs et négatifs"
               + (", avec quelques indices de coordination à vérifier" if coordination else "")
               + " : rien ne désigne un camp en particulier.")
lignes.append(f"**En résumé.** {verdict} Un signal n'est pas une preuve : chacun a une explication innocente "
              "possible, et la décision revient à la relecture humaine.")
display(Markdown("## Conclusion\n\n" + "\n\n".join(lignes)))
