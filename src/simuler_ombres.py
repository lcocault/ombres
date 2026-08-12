#!/usr/bin/env python3
"""
simuler_ombres.py — Simulation des zones d'ombre d'un potager.

Usage:
    python src/simuler_ombres.py --latitude 48.8566 --longitude 2.3522 \
        --date 2026-08-12 --heures 6 20

Options:
    --latitude    Latitude GPS du potager (degrés décimaux)
    --longitude   Longitude GPS du potager (degrés décimaux)
    --date        Date de simulation (YYYY-MM-DD)
    --heures      Plage horaire (heure_debut heure_fin, ex: 6 20)
    --mns         Chemin vers le fichier CSV MNS  (défaut: data/mns_potager.csv)
    --resolution  Résolution en mètres par pixel  (défaut: 1.0)
    --output      Dossier de sortie               (défaut: results)
"""

import argparse
import csv
import logging
import math
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape
from PIL import Image, ImageDraw
from skyfield.api import N, E, load, wgs84

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MNS helpers
# ---------------------------------------------------------------------------

def load_mns(csv_path: str, resolution: float = 1.0) -> tuple[np.ndarray, float, float, float, float]:
    """Charge le MNS CSV et retourne (grid_altitude, x_min, y_min, dx, dy)."""
    df = pd.read_csv(csv_path)
    required = {"X", "Y", "altitude"}
    if not required.issubset(df.columns):
        raise ValueError(f"Le fichier MNS doit contenir les colonnes: {required}")

    df["X"] = df["X"] * resolution
    df["Y"] = df["Y"] * resolution

    x_min, x_max = df["X"].min(), df["X"].max()
    y_min, y_max = df["Y"].min(), df["Y"].max()

    cols = round((x_max - x_min) / resolution) + 1
    rows = round((y_max - y_min) / resolution) + 1

    grid = np.zeros((rows, cols), dtype=float)
    for _, row in df.iterrows():
        c = round((row["X"] - x_min) / resolution)
        r = round((row["Y"] - y_min) / resolution)
        if 0 <= r < rows and 0 <= c < cols:
            grid[r, c] = row["altitude"]

    return grid, x_min, y_min, resolution, resolution


# ---------------------------------------------------------------------------
# Solar position (azimuth / altitude)
# ---------------------------------------------------------------------------

def _sun_position_approx(lat: float, lon: float, dt: datetime) -> tuple[float, float]:
    """
    Calcul approché de la position du soleil (précision ~1°) sans accès réseau.
    Algorithme basé sur les formules de Jean Meeus (Astronomical Algorithms).
    dt doit être timezone-aware (UTC).
    """
    jd = dt.timestamp() / 86400.0 + 2440587.5
    n = jd - 2451545.0  # J2000.0

    L = math.radians((280.46 + 0.9856474 * n) % 360)
    g = math.radians((357.528 + 0.9856003 * n) % 360)
    lam = L + math.radians(1.915 * math.sin(g) + 0.020 * math.sin(2 * g))
    eps = math.radians(23.439 - 0.0000004 * n)

    sin_lam = math.sin(lam)
    dec = math.asin(math.sin(eps) * sin_lam)
    ra = math.atan2(math.cos(eps) * sin_lam, math.cos(lam))

    theta0 = math.radians((280.46061837 + 360.98564736629 * n) % 360)
    ha = theta0 + math.radians(lon) - ra

    lat_r = math.radians(lat)
    sin_alt = (math.sin(lat_r) * math.sin(dec)
               + math.cos(lat_r) * math.cos(dec) * math.cos(ha))
    alt = math.degrees(math.asin(max(-1.0, min(1.0, sin_alt))))

    cos_alt_r = math.cos(math.asin(max(-1.0, min(1.0, sin_alt))))
    if abs(cos_alt_r) < 1e-10:
        return 0.0, alt
    cos_az = (math.sin(dec) - math.sin(lat_r) * sin_alt) / (math.cos(lat_r) * cos_alt_r)
    cos_az = max(-1.0, min(1.0, cos_az))
    az = math.degrees(math.acos(cos_az))
    if math.sin(ha) > 0:
        az = 360 - az

    return az, alt


def get_sun_position(lat: float, lon: float, dt: datetime) -> tuple[float, float]:
    """
    Retourne (azimut_deg, altitude_deg) du soleil pour la position et
    l'instant donnés.  dt doit être un datetime timezone-aware (UTC).

    Utilise skyfield (éphémérides DE421) si disponible, sinon bascule sur
    un calcul approché.
    """
    try:
        ts = load.timescale()
        t = ts.from_datetime(dt)
        eph = load("de421.bsp")
        location = wgs84.latlon(lat * N, lon * E)
        astrometric = location.at(t).observe(eph["sun"]).apparent()
        alt, az, _ = astrometric.altaz()
        return az.degrees, alt.degrees
    except OSError:
        log.warning("Éphémérides DE421 non disponibles — utilisation du calcul approché.")
        return _sun_position_approx(lat, lon, dt)


# ---------------------------------------------------------------------------
# Ray-casting shadow algorithm
# ---------------------------------------------------------------------------

def compute_shadow_map(
    grid: np.ndarray,
    azimuth_deg: float,
    altitude_deg: float,
    resolution: float,
) -> np.ndarray:
    """
    Calcule la carte d'ombre (True = ombré) par ray casting.

    Pour chaque cellule, on suit un rayon en direction opposée au soleil et
    on vérifie si une cellule plus haute bloque la lumière.
    """
    rows, cols = grid.shape
    shadow = np.zeros((rows, cols), dtype=bool)

    if altitude_deg <= 0:
        # Soleil sous l'horizon → tout est dans l'ombre
        shadow[:] = True
        return shadow

    # Direction du rayon vers le soleil (vecteur horizontal unitaire)
    az_rad = math.radians(azimuth_deg)
    # Déplacement d'un pas vers le soleil dans la grille
    # (azimut mesuré depuis le Nord, sens horaire)
    step_col = math.sin(az_rad)   # Est positif → colonne croissante
    step_row = -math.cos(az_rad)  # Nord positif → ligne décroissante
    tan_alt = math.tan(math.radians(altitude_deg))

    # Calculé une seule fois hors de la boucle interne
    max_steps = int(math.hypot(rows, cols)) + 1
    step_dist = resolution * math.hypot(step_row, step_col)

    for r in range(rows):
        for c in range(cols):
            cell_alt = grid[r, c]
            # Suivre le rayon PAS À PAS vers le soleil
            for step in range(1, max_steps):
                tr = r + step * step_row
                tc = c + step * step_col
                ri, ci = int(round(tr)), int(round(tc))
                if ri < 0 or ri >= rows or ci < 0 or ci >= cols:
                    break
                obstacle_alt = grid[ri, ci]
                # Hauteur du rayon solaire à cette distance
                ray_height = cell_alt + step * step_dist * tan_alt
                if obstacle_alt > ray_height:
                    shadow[r, c] = True
                    break

    return shadow


# ---------------------------------------------------------------------------
# Image generation
# ---------------------------------------------------------------------------

def shadow_to_image(shadow: np.ndarray, hour_label: str, output_path: str) -> None:
    """Génère une image PNG de la carte d'ombre."""
    rows, cols = shadow.shape
    # Soleil = jaune, ombre = bleu foncé
    rgb = np.where(shadow[:, :, np.newaxis], [30, 50, 120], [255, 220, 50]).astype(np.uint8)
    img = Image.fromarray(rgb, mode="RGB")
    # Scale up for visibility
    scale = max(1, 40 // max(rows, cols, 1))
    if scale > 1:
        img = img.resize((cols * scale, rows * scale), Image.NEAREST)
    draw = ImageDraw.Draw(img)
    draw.text((4, 4), hour_label, fill=(255, 255, 255))
    img.save(output_path)


# ---------------------------------------------------------------------------
# Matplotlib chart
# ---------------------------------------------------------------------------

def generate_sunshine_chart(hours: list[int], sunshine_pct: list[float], output_path: str) -> None:
    """Génère le graphique de l'évolution de l'ensoleillement."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(hours, sunshine_pct, marker="o", color="#f5a623", linewidth=2)
    ax.fill_between(hours, sunshine_pct, alpha=0.2, color="#f5a623")
    ax.set_xlabel("Heure")
    ax.set_ylabel("Surface ensoleillée (%)")
    ax.set_title("Évolution de l'ensoleillement dans la journée")
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Agronomic analysis
# ---------------------------------------------------------------------------

AGRO_RULES = [
    (90, "☀️ Ensoleillement optimal — idéal pour tomates, poivrons, courgettes."),
    (70, "🌤️ Bon ensoleillement — convient aux haricots, salades et carottes."),
    (50, "⛅ Ensoleillement partiel — adapté aux épinards et radis."),
    (0,  "🌥️ Faible ensoleillement — recommandé pour les plantes d'ombre (cresson, mâche)."),
]


def agronomic_note(sunshine_pct: float) -> str:
    for threshold, note in AGRO_RULES:
        if sunshine_pct >= threshold:
            return f"{sunshine_pct:.1f}% du potager est ensoleillé → {note}"
    return f"{sunshine_pct:.1f}% du potager est ensoleillé."


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8">
  <title>Rapport d'ombres — {{ date }}</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 2em; background: #fafafa; color: #333; }
    h1 { color: #2c7a2c; }
    h2 { color: #444; border-bottom: 2px solid #ddd; padding-bottom: .3em; }
    .card-grid { display: flex; flex-wrap: wrap; gap: 1em; }
    .card { background: white; border: 1px solid #ddd; border-radius: 8px; padding: 1em;
            box-shadow: 0 2px 4px rgba(0,0,0,.08); text-align: center; }
    .card img { max-width: 200px; image-rendering: pixelated; }
    .agro { background: #fff8e1; border-left: 4px solid #f5a623; padding: .5em 1em;
            margin: .4em 0; border-radius: 4px; }
    .chart img { max-width: 800px; }
  </style>
</head>
<body>
  <h1>🌿 Rapport des zones d'ombre — {{ date }}</h1>
  <p><strong>Coordonnées :</strong> {{ latitude }}°N, {{ longitude }}°E</p>

  <h2>📊 Évolution de l'ensoleillement</h2>
  <div class="chart"><img src="{{ chart_path }}" alt="Graphique ensoleillement"></div>

  <h2>🗺️ Cartes horaires des ombres</h2>
  <div class="card-grid">
  {% for hour, img_path, agro in hours_data %}
    <div class="card">
      <p><strong>{{ "%02d:00"|format(hour) }}</strong></p>
      <img src="{{ img_path }}" alt="Ombre {{ hour }}h">
      <p class="agro">{{ agro }}</p>
    </div>
  {% endfor %}
  </div>
</body>
</html>
"""


def generate_html_report(
    output_dir: str,
    sim_date: str,
    latitude: float,
    longitude: float,
    hours_data: list[tuple[int, str, str]],
    chart_path: str,
) -> str:
    env = Environment(loader=FileSystemLoader("/"), autoescape=select_autoescape(["html"]))
    template = env.from_string(HTML_TEMPLATE)
    html = template.render(
        date=sim_date,
        latitude=latitude,
        longitude=longitude,
        hours_data=hours_data,
        chart_path=chart_path,
    )
    report_path = os.path.join(output_dir, "rapport_ombres.html")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return report_path


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def save_csv(output_dir: str, rows: list[dict]) -> str:
    csv_path = os.path.join(output_dir, "ombres_par_heure.csv")
    if not rows:
        return csv_path
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Simule les zones d'ombre d'un potager.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--latitude", type=float, required=True, help="Latitude GPS (degrés)")
    parser.add_argument("--longitude", type=float, required=True, help="Longitude GPS (degrés)")
    parser.add_argument("--date", type=str, required=True, help="Date (YYYY-MM-DD)")
    parser.add_argument("--heures", type=int, nargs=2, metavar=("DEBUT", "FIN"),
                        default=[6, 20], help="Plage horaire [début fin]")
    parser.add_argument("--mns", type=str, default="data/mns_potager.csv",
                        help="Chemin vers le fichier MNS CSV")
    parser.add_argument("--resolution", type=float, default=1.0,
                        help="Résolution en mètres par pixel (défaut: 1.0)")
    parser.add_argument("--output", type=str, default="results",
                        help="Dossier de sortie (défaut: results)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # Resolve paths relative to repo root (the script may be called from anywhere)
    repo_root = Path(__file__).resolve().parent.parent
    mns_path = Path(args.mns) if Path(args.mns).is_absolute() else repo_root / args.mns
    output_dir = Path(args.output) if Path(args.output).is_absolute() else repo_root / args.output
    cartes_dir = output_dir / "cartes"
    cartes_dir.mkdir(parents=True, exist_ok=True)

    log.info("Chargement du MNS: %s", mns_path)
    grid, x_min, y_min, dx, dy = load_mns(str(mns_path), args.resolution)
    log.info("Grille MNS: %d×%d cellules (résolution %.2f m)", grid.shape[0], grid.shape[1], args.resolution)

    sim_date = datetime.strptime(args.date, "%Y-%m-%d").date()
    heure_debut, heure_fin = args.heures
    hours = list(range(heure_debut, heure_fin + 1))

    csv_rows: list[dict] = []
    sunshine_pct_list: list[float] = []
    html_hours_data: list[tuple[int, str, str]] = []

    for h in hours:
        dt = datetime(sim_date.year, sim_date.month, sim_date.day, h, 0, 0, tzinfo=timezone.utc)
        log.info("Calcul ombre à %02d:00 UTC...", h)

        az, alt = get_sun_position(args.latitude, args.longitude, dt)
        log.info("  Soleil → azimut=%.1f°, altitude=%.1f°", az, alt)

        shadow = compute_shadow_map(grid, az, alt, args.resolution)
        total = shadow.size
        shadowed = shadow.sum()
        sunlit = total - shadowed
        pct = 100.0 * sunlit / total if total > 0 else 0.0
        sunshine_pct_list.append(pct)

        img_filename = f"ombre_{h:02d}h.png"
        img_path = cartes_dir / img_filename
        shadow_to_image(shadow, f"{h:02d}:00", str(img_path))
        log.info("  Image sauvegardée: %s", img_path)

        agro = agronomic_note(pct)
        log.info("  Analyse: %s", agro)

        csv_rows.append({
            "heure": f"{h:02d}:00",
            "azimut_soleil": round(az, 2),
            "altitude_soleil": round(alt, 2),
            "surface_ensoleillée_pct": round(pct, 2),
            "analyse_agronomique": agro,
        })
        # Paths relative to output_dir for HTML portability
        html_hours_data.append((h, f"cartes/{img_filename}", agro))

    # Chart
    chart_filename = "evolution_ensoleillement.png"
    chart_path = output_dir / chart_filename
    generate_sunshine_chart(hours, sunshine_pct_list, str(chart_path))
    log.info("Graphique sauvegardé: %s", chart_path)

    # CSV
    csv_path = save_csv(str(output_dir), csv_rows)
    log.info("CSV sauvegardé: %s", csv_path)

    # HTML report
    report_path = generate_html_report(
        str(output_dir),
        args.date,
        args.latitude,
        args.longitude,
        html_hours_data,
        chart_filename,
    )
    log.info("Rapport HTML: %s", report_path)
    log.info("✅ Simulation terminée. Résultats dans: %s", output_dir)


if __name__ == "__main__":
    main()
