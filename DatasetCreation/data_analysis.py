import os
import pandas as pd
from datetime import datetime
from sunpy.net import Fido, attrs as a

# =============================
# 1. PARAMÈTRES UTILISATEUR
# =============================

# Fichier d'entrée
input_csv = "cmes_lz_20120101_20151231.csv"

start_time = datetime(2012, 1, 1)
end_time   = datetime(2015, 12, 31)

if not os.path.exists(input_csv):
    raise FileNotFoundError(f"❌ Fichier introuvable : {input_csv}")

# =============================
# 2. LECTURE DU FICHIER CACTUS
# =============================

df = pd.read_csv(input_csv, parse_dates=["t0"])
print(f"✅ Fichier chargé : {len(df)} événements CME")

# Détermination automatique de la période couverte
start_time = df["t0"].min()
end_time = df["t0"].max()
print(f"Période couverte : {start_time} → {end_time}")

# =============================
# 3. STATISTIQUES DE BASE
# =============================

print("\n=== STATISTIQUES CACTUS ===")

n_cme = len(df)
n_days = (end_time - start_time).days + 1
mean_per_day = n_cme / n_days

print(f"Nombre total de CME : {n_cme}")
print(f"Nombre de jours observés : {n_days}")
print(f"Nombre moyen de CME / jour : {mean_per_day:.2f}")

# Distribution des vitesses
print("\nDistribution des vitesses (km/s) :")
print(df["v_km_s"].describe(percentiles=[0.1, 0.5, 0.9]))

# Comptage par "halo" (si renseigné)
if "halo" in df.columns:
    print("\nProportion de CME halo (complètes) :")
    print(df["halo"].value_counts(dropna=False))

# Histogramme rapide
try:
    import matplotlib.pyplot as plt
    plt.hist(df["v_km_s"].dropna(), bins=20)
    plt.xlabel("Vitesse (km/s)")
    plt.ylabel("Nombre de CME")
    plt.title("Distribution des vitesses CME - CACTus")
    plt.show()
except ImportError:
    print("(Matplotlib non installé — skipping plot)")

# =============================
# 4. VÉRIFICATION DES IMAGES LASCO-C2 DISPONIBLES
# =============================

print("\n=== DISPONIBILITÉ DES IMAGES LASCO-C2 ===")
print(f"Recherche d’images LASCO-C2 entre {start_time} et {end_time} (sans téléchargement)...")

query = Fido.search(
    a.Time(start_time, end_time),
    a.Instrument.lasco,
    a.Detector.c2,
    a.Level(1)
)

# Fido retourne un QueryResponse
if len(query) == 0:
    print("❌ Aucune image LASCO-C2 trouvée pour cette période.")
else:
    n_results = len(query[0])
    providers = query[0].provider.unique()
    print(f"✅ {n_results} images LASCO-C2 disponibles.")
    print(f"Provenance : {providers}")

# =============================
# 5. SYNTHÈSE GLOBALE
# =============================

print("\n=== SYNTHÈSE ===")
print(f"• {n_cme} CME détectées entre {start_time.date()} et {end_time.date()}")
print(f"• {mean_per_day:.2f} CME/jour en moyenne")
if len(query) > 0:
    print(f"• {n_results} images LASCO-C2 disponibles dans la même période")
    ratio = n_results / n_cme if n_cme > 0 else float('nan')
    print(f"• Ratio images / CME ≈ {ratio:.1f}")
else:
    print("• Pas d’images LASCO-C2 trouvées pour comparer.")

print("\n✅ Analyse terminée.")
