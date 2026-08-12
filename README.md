# ombres

Simule les zones d'ombre d'un potager en fonction de la date, de l'heure et d'un Modèle Numérique de Surface (MNS).

## Structure du projet

```
potager_ombres/
├── data/
│   └── mns_potager.csv        # Matrice MNS (colonnes X, Y, altitude)
├── results/
│   ├── cartes/                # Images PNG des zones d'ombre (une par heure)
│   ├── rapport_ombres.html    # Rapport HTML interactif
│   └── ombres_par_heure.csv   # Tableau récapitulatif
└── src/
    └── simuler_ombres.py      # Script principal
```

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

```bash
python src/simuler_ombres.py \
    --latitude 48.8566 \
    --longitude 2.3522 \
    --date 2026-08-12 \
    --heures 6 20
```

### Options disponibles

| Option | Description | Défaut |
|---|---|---|
| `--latitude` | Latitude GPS du potager (degrés décimaux) | *obligatoire* |
| `--longitude` | Longitude GPS du potager (degrés décimaux) | *obligatoire* |
| `--date` | Date de simulation (`YYYY-MM-DD`) | *obligatoire* |
| `--heures` | Plage horaire (début fin) | `6 20` |
| `--mns` | Chemin vers le fichier MNS CSV | `data/mns_potager.csv` |
| `--resolution` | Résolution en mètres par pixel | `1.0` |
| `--output` | Dossier de sortie | `results` |

### Exemple avec résolution personnalisée

```bash
python src/simuler_ombres.py \
    --latitude 45.7640 \
    --longitude 4.8357 \
    --date 2026-06-21 \
    --heures 8 18 \
    --resolution 0.5 \
    --mns data/mns_potager.csv \
    --output results
```

## Format du fichier MNS

Le fichier CSV doit contenir les colonnes `X`, `Y` et `altitude` (en mètres) :

```csv
X,Y,altitude
0,0,0.0
1,0,0.0
2,0,2.5
```

Les points représentent la hauteur de chaque obstacle (haie, arbre, bâtiment…) sur la grille du potager.

## Sorties

- **`results/cartes/ombre_HHh.png`** — Carte de l'ombre à chaque heure (jaune = ensoleillé, bleu = ombré).
- **`results/ombres_par_heure.csv`** — Tableau avec azimut solaire, altitude solaire, % ensoleillé et analyse agronomique.
- **`results/rapport_ombres.html`** — Rapport complet avec cartes, graphique et recommandations de culture.

## Fonctionnement

1. La position exacte du soleil (azimut, altitude) est calculée via la bibliothèque **skyfield** (éphémérides DE421).
2. Un algorithme de **ray casting** détermine, pour chaque cellule de la grille, si un obstacle se trouve sur le trajet du rayon solaire.
3. Les résultats sont agrégés pour produire les images, le CSV et le rapport HTML.
