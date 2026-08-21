"""
compile_simulations.py
----------------------
Compile les résultats de 560 fichiers de simulation Monte Carlo.

Nommage des fichiers :
    log_big_4_v_{density}_l_{lambda}_z_{LZ}_pol_{pol}.txt

Sorties :
    - 1 CSV unique avec toutes les données (une ligne par fichier)
    - 8 CSV (2 par valeur de Lambda) :
        * trié par polarisation puis Z croissant
        * trié par polarisation puis density croissante
"""

import os
import re
import csv
import glob
from collections import defaultdict

# ── Paramètres de simulation ──────────────────────────────────────────────────
polar_inputs = {
    "1000": (1, 0, 0, 0),
    "1001": (1, 0, 0, 1),
    "1010": (1, 0, 1, 0),
    "1100": (1, 1, 0, 0),
}
density_values = [20, 35, 47, 65, 124]
Lambda_values  = [0.55, 1.55, 4, 9.8]
LZ_values      = [100, 500, 1000, 2000, 3000, 4000, 5000]

# ── Colonnes du CSV de sortie ─────────────────────────────────────────────────
COLUMNS = [
    "filename",
    "density", "lambda", "LZ", "polarisation",
    # Bilan global
    "R_photons", "R_frac",
    "T_photons", "T_frac",
    "A_photons", "A_frac",
    "detector_photons", "detector_rate", "detector_pct",
    # Réflexion – Stokes brut
    "refl_I", "refl_Q", "refl_U", "refl_V",
    # Réflexion – Stokes normalisé
    "refl_I_norm", "refl_Q_norm", "refl_U_norm", "refl_V_norm",
    # Réflexion – degrés de polarisation
    "refl_pol_total", "refl_pol_lin", "refl_pol_circ",
    # Transmission – Stokes brut
    "trans_I", "trans_Q", "trans_U", "trans_V",
    # Transmission – Stokes normalisé
    "trans_I_norm", "trans_Q_norm", "trans_U_norm", "trans_V_norm",
    # Transmission – degrés de polarisation
    "trans_pol_total", "trans_pol_lin", "trans_pol_circ",
    # Détecteur – Stokes brut
    "det_I", "det_Q", "det_U", "det_V",
    # Détecteur – Stokes normalisé
    "det_I_norm", "det_Q_norm", "det_U_norm", "det_V_norm",
    # Détecteur – degrés de polarisation
    "det_pol_total", "det_pol_lin", "det_pol_circ",
]


# ── Parsing ───────────────────────────────────────────────────────────────────

def parse_filename(fname):
    """Extrait density, lambda, LZ, pol depuis le nom de fichier."""
    base = os.path.basename(fname)
    m = re.match(
        r"log_big_4_v_(\d+)_l_([\d.]+)_z_(\d+)_pol_(\d+)\.txt",
        base
    )
    if not m:
        return None
    return {
        "density":      int(m.group(1)),
        "lambda":       float(m.group(2)),
        "LZ":           int(m.group(3)),
        "polarisation": m.group(4),
    }


def _float(s):
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def parse_file(path):
    """Parse un fichier de simulation et renvoie un dict de valeurs."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    d = {}

    # ── Bilan global ──────────────────────────────────────────────────────────
    # Réflexion (R)
    m = re.search(r"Réflexion \(R\)\s*:\s*([\d.]+)\s*\(\s*([\d.]+)\s*\)", text)
    if m:
        d["R_photons"], d["R_frac"] = _float(m.group(1)), _float(m.group(2))

    # Transmission (T)
    m = re.search(r"Transmission \(T\)\s*:\s*([\d.]+)\s*\(\s*([\d.]+)\s*\)", text)
    if m:
        d["T_photons"], d["T_frac"] = _float(m.group(1)), _float(m.group(2))

    # Absorption (A)
    m = re.search(r"Absorption \(A\)\s*:\s*([\d.]+)\s*\(\s*([\d.]+)\s*\)", text)
    if m:
        d["A_photons"], d["A_frac"] = _float(m.group(1)), _float(m.group(2))

    # Détecteur global
    m = re.search(r"Photons collectés\s*:\s*([\d.]+)", text)
    if m:
        d["detector_photons"] = _float(m.group(1))
    m = re.search(r"Taux de collecte\s*:\s*([\d.]+)", text)
    if m:
        d["detector_rate"] = _float(m.group(1))
    m = re.search(r"Pourcentage collecté\s*:\s*([\d.]+)", text)
    if m:
        d["detector_pct"] = _float(m.group(1))

    # ── Sections de polarisation ──────────────────────────────────────────────
    # On découpe le texte en trois blocs : RÉFLEXION / TRANSMISSION / DÉTECTEUR
    # dans la partie "RÉSULTATS DE POLARISATION"
    pol_section = re.split(r"RÉSULTATS DE POLARISATION", text, maxsplit=1)
    if len(pol_section) < 2:
        return d
    pol_text = pol_section[1]

    # Sépare les trois sous-sections
    parts = re.split(r"-{10,}\s*(RÉFLEXION|TRANSMISSION|DÉTECTEUR)\s*-{10,}", pol_text)
    # parts = [avant, label1, contenu1, label2, contenu2, ...]
    sections = {}
    for i in range(1, len(parts) - 1, 2):
        label   = parts[i].strip()
        content = parts[i + 1]
        sections[label] = content

    def extract_stokes_brut(content, prefix):
        """Stokes bruts : I Q U V"""
        m = re.search(
            r"I\s*=\s*([-\d.]+).*?Q\s*=\s*([-\d.]+).*?U\s*=\s*([-\d.]+).*?V\s*=\s*([-\d.]+)",
            content, re.DOTALL
        )
        if m:
            d[f"{prefix}_I"] = _float(m.group(1))
            d[f"{prefix}_Q"] = _float(m.group(2))
            d[f"{prefix}_U"] = _float(m.group(3))
            d[f"{prefix}_V"] = _float(m.group(4))

    def extract_stokes_norm(content, prefix):
        """Stokes normalisés : deuxième bloc I Q U V"""
        matches = list(re.finditer(
            r"I\s*=\s*([-\d.]+).*?Q\s*=\s*([-\d.]+).*?U\s*=\s*([-\d.]+).*?V\s*=\s*([-\d.]+)",
            content, re.DOTALL
        ))
        if len(matches) >= 2:
            m = matches[1]
            d[f"{prefix}_I_norm"] = _float(m.group(1))
            d[f"{prefix}_Q_norm"] = _float(m.group(2))
            d[f"{prefix}_U_norm"] = _float(m.group(3))
            d[f"{prefix}_V_norm"] = _float(m.group(4))

    def extract_pol_degrees(content, prefix):
        m = re.search(r"Polarisation totale\s*=\s*([\d.]+)", content)
        if m:
            d[f"{prefix}_pol_total"] = _float(m.group(1))
        m = re.search(r"Polarisation lin[eé]aire\s*=\s*([\d.]+)", content)
        if m:
            d[f"{prefix}_pol_lin"] = _float(m.group(1))
        m = re.search(r"Polarisation circulaire\s*=\s*([\d.]+)", content)
        if m:
            d[f"{prefix}_pol_circ"] = _float(m.group(1))

    # Réflexion
    if "RÉFLEXION" in sections:
        c = sections["RÉFLEXION"]
        extract_stokes_brut(c, "refl")
        extract_stokes_norm(c, "refl")
        extract_pol_degrees(c, "refl")

    # Transmission
    if "TRANSMISSION" in sections:
        c = sections["TRANSMISSION"]
        extract_stokes_brut(c, "trans")
        extract_stokes_norm(c, "trans")
        extract_pol_degrees(c, "trans")

    # Détecteur (dans la section polarisation)
    if "DÉTECTEUR" in sections:
        c = sections["DÉTECTEUR"]
        extract_stokes_brut(c, "det")
        extract_stokes_norm(c, "det")
        extract_pol_degrees(c, "det")

    return d


# ── Pipeline principal ────────────────────────────────────────────────────────

def collect_all_rows(input_dir="."):
    """Parcourt tous les fichiers .txt et renvoie une liste de dicts."""
    pattern = os.path.join(input_dir, "log_big_4_v_*_l_*_z_*_pol_*.txt")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"⚠  Aucun fichier trouvé dans : {os.path.abspath(input_dir)}")
        return []

    rows = []
    errors = []
    for path in files:
        params = parse_filename(path)
        if params is None:
            errors.append(path)
            continue
        try:
            data = parse_file(path)
        except Exception as e:
            errors.append(f"{path} ({e})")
            continue

        row = {"filename": os.path.basename(path)}
        row.update(params)
        row.update(data)
        rows.append(row)

    print(f"✓  {len(rows)} fichiers compilés, {len(errors)} erreurs.")
    if errors:
        print("  Fichiers en erreur :")
        for e in errors:
            print(f"    {e}")
    return rows


def write_csv(rows, path, fieldnames=None):
    if fieldnames is None:
        fieldnames = COLUMNS
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  → {path}  ({len(rows)} lignes)")


def main(input_dir=".", output_dir="output_csv"):
    os.makedirs(output_dir, exist_ok=True)

    rows = collect_all_rows(input_dir)
    if not rows:
        return

    # ── 1. CSV unique ─────────────────────────────────────────────────────────
    all_sorted = sorted(rows, key=lambda r: (
        r["lambda"], r["polarisation"], r["LZ"], r["density"]
    ))
    write_csv(all_sorted, os.path.join(output_dir, "all_results.csv"))

    # ── 2. 8 CSV par Lambda (2 tris) ─────────────────────────────────────────
    # Regroupe par lambda
    by_lambda = defaultdict(list)
    for r in rows:
        by_lambda[r["lambda"]].append(r)

    for lam, lrows in sorted(by_lambda.items()):
        lam_str = str(lam).replace(".", "_")

        # Tri 1 : par polarisation puis Z croissant
        sorted_z = sorted(lrows, key=lambda r: (r["polarisation"], r["LZ"], r["density"]))
        write_csv(
            sorted_z,
            os.path.join(output_dir, f"lambda_{lam_str}_sort_z.csv")
        )

        # Tri 2 : par polarisation puis density croissante
        sorted_d = sorted(lrows, key=lambda r: (r["polarisation"], r["density"], r["LZ"]))
        write_csv(
            sorted_d,
            os.path.join(output_dir, f"lambda_{lam_str}_sort_density.csv")
        )

    print("\nTerminé.")


if __name__ == "__main__":
    import sys
    input_dir  = sys.argv[1] if len(sys.argv) > 1 else "."
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output_csv"
    main(input_dir, output_dir)