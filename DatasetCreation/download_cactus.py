import sunpy.map
from sunpy.net import Fido, attrs as a
from datetime import datetime, timedelta
import pandas as pd
import requests
from scipy.io import readsav  # (restera inutilisé ici, mais conservé pour compatibilité)
import os
from io import StringIO

# =============================
# 1. PARAMÈTRES UTILISATEUR
# =============================

# Exemple : une période d'un mois 
start_time = datetime(2022, 1, 1)
end_time   = datetime(2022, 12, 31)

# =============================
# 2. RÉCUPÉRATION DU CATALOGUE CACTUS (level‑zero)
# =============================

# URL du fichier texte contenant toutes les CME détectées (level‑zero)
cme_lz_url = "https://www.sidc.be/cactus/catalog/LASCO/2_5_0/cme_qkl.txt"

print("Téléchargement du fichier cme_lz.txt...")
response = requests.get(cme_lz_url)
response.raise_for_status()

# Noms de colonnes selon la description du fichier
column_names = ["CME_ID", "t0", "dt0_h", "pa_deg", "da_deg",
                "v_km_s", "dv_km_s", "minv_km_s", "maxv_km_s", "halo"]

# Lire le contenu dans un DataFrame en ignorant les lignes de commentaires (#) et en utilisant '|' comme séparateur
df_cme = pd.read_csv(
    StringIO(response.text),
    sep="|",
    comment="#",
    header=None,
    names=column_names,
    skipinitialspace=True,
    dtype={
        "CME_ID": str,
        "dt0_h": float,
        "pa_deg": float,
        "da_deg": float,
        "v_km_s": float,
        "dv_km_s": float,
        "minv_km_s": float,
        "maxv_km_s": float,
        "halo": str,
    }
)

# Nettoyer les colonnes (supprimer les espaces superflus)
df_cme["CME_ID"] = df_cme["CME_ID"].str.strip()
df_cme["halo"]   = df_cme["halo"].str.strip()

# Convertir la colonne t0 en datetime
df_cme["t0"] = pd.to_datetime(df_cme["t0"].str.strip(), format="%Y/%m/%d %H:%M")

print("Aperçu des données CME level-zero téléchargées :")
print(df_cme.head())

# =============================
# 4. FILTRAGE DES CME SUR LA PÉRIODE SOUHAITÉE
# =============================

# Sélectionner uniquement les CME dont t0 est entre start_time et end_time (intervalle semi-ouvert)
mask = (df_cme["t0"] >= start_time) & (df_cme["t0"] < end_time)
filtered_cmes = df_cme.loc[mask]

print(f"CME level-zero détectées entre {start_time} et {end_time} : {len(filtered_cmes)} événement(s)")
print(filtered_cmes[["CME_ID", "t0", "pa_deg", "da_deg", "v_km_s", "minv_km_s", "maxv_km_s", "halo"]])

# Optionnel : sauvegarder les résultats filtrés dans un CSV
if not filtered_cmes.empty:
    output_file = f"cmes_lz_{start_time.strftime('%Y%m%d')}_{end_time.strftime('%Y%m%d')}.csv"
    filtered_cmes.to_csv(output_file, index=False)
    print(f"Données sauvegardées dans {output_file}")
else:
    print("Aucun événement CME détecté dans cette période.")
