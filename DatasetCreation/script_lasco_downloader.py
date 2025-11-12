#!/usr/bin/env python3
"""
Télécharge et convertit toutes les images LASCO-C2 de 2022
en PNG 512x512 avec barre de progression et reprise possible.
"""

import os
import argparse
from datetime import datetime, timedelta
from sunpy.net import Fido, attrs as a
from tqdm import tqdm

# =====================================================
# 1. PARAMÈTRES UTILISATEUR
# =====================================================

DATA_DIR = "/Users/hippolytehilgers/Desktop/ROSACE/Project_ROSACE/WP2/Dataset/data_lasco_c2"
PNG_DIR  = "/Users/hippolytehilgers/Desktop/ROSACE/Project_ROSACE/WP2/Dataset/data_lasco_c2_png"

# Ne pas créer les dossiers, s'ils n'existent pas cela signifie qu'on veut une erreur
#os.makedirs(DATA_DIR, exist_ok=True)
#os.makedirs(PNG_DIR, exist_ok=True)

from PIL import Image
import numpy as np
import sunpy.map
import os
from datetime import datetime
import re

# =====================================================
# 1.5. FONCTION DE CONVERSION .fts → .png
# =====================================================

def fts_to_png(input_path, output_dir, size=(512, 512), contrast_clip=(1, 99)):
    """
    Convertit un fichier LASCO .fts/.fits en PNG 512x512.
    Le nom du PNG est basé sur la combinaison DATE-OBS + TIME-OBS.
    """

    try:
        # Lecture du FITS via SunPy
        smap = sunpy.map.Map(input_path)
        data = smap.data.astype(float)
        header = smap.meta

        # Lecture des champs DATE-OBS et TIME-OBS
        date_part = header.get('DATE-OBS', '')
        time_part = header.get('TIME-OBS', '')

        if date_part and time_part:
            date_str_full = f"{date_part}T{time_part}"
        elif date_part:  # parfois TIME-OBS manquant, fallback
            date_str_full = date_part
        else:
            # fallback : nom du fichier
            date_str_full = os.path.splitext(os.path.basename(input_path))[0]

        # Conversion en format standard YYYYMMDD_HHMMSS
        try:
            date_obj = datetime.strptime(date_str_full, "%Y-%m-%dT%H:%M:%S")
            date_str = date_obj.strftime("%Y%m%d_%H%M%S")
        except Exception:
            date_str = re.sub(r'[^0-9]', '_', date_str_full)

        # Nettoyage des caractères spéciaux (sécurité)
        date_str = re.sub(r'[^0-9A-Za-z_]', '_', date_str)

        # Normalisation de l'image
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        vmin, vmax = np.percentile(data, contrast_clip)
        data = np.clip(data, vmin, vmax)
        data = (255 * (data - vmin) / (vmax - vmin)).astype(np.uint8)

        # Conversion en PNG 512x512
        img = Image.fromarray(data)
        img = img.resize(size, Image.Resampling.LANCZOS)

        # Nom de sortie avec la date/heure incluse
        os.makedirs(output_dir, exist_ok=True)
        filename = f"LASCOC2_{date_str}.png"
        output_path = os.path.join(output_dir, filename)

        img.save(output_path)

        return output_path

    except Exception as e:
        print(f"❌ Erreur pour {input_path} : {e}")
        return None



# =====================================================
# 2. FONCTION PRINCIPALE
# =====================================================

def download_and_convert_lasco_c2(start_time, end_time, step_days=30):
    """
    Télécharge et convertit les images LASCO-C2 entre start_time et end_time.
    Le téléchargement est fait par fenêtres mensuelles pour éviter les timeouts.
    """

    current = start_time
    total_converted = 0

    while current < end_time:
        window_end = min(current + timedelta(days=step_days), end_time)
        print(f"\n🔍 Recherche LASCO-C2 du {current.date()} au {window_end.date()} ...")

        query = Fido.search(
            a.Time(current, window_end),
            a.Instrument.lasco,
            a.Detector.c2
        )

        if len(query) == 0:
            print("⚠️  Aucune image trouvée pour cette période.")
            current = window_end
            continue

        n_results = len(query[0])
        print(f"➡️  {n_results} fichiers trouvés. Téléchargement en cours...")

        # Téléchargement avec barre de progression
        downloaded_files = []
        for i in tqdm(range(n_results), desc="Téléchargement FITS", ncols=80):
            try:
                file_list = Fido.fetch(query[0][i:i+1], progress=False, path=os.path.join(DATA_DIR, "{file}"))
                if file_list:
                    downloaded_files.append(file_list[0])
            except Exception as e:
                print(f"Erreur sur téléchargement {i}: {e}")

        print(f"✅ {len(downloaded_files)} fichiers téléchargés. Conversion en PNG...")

        # Conversion avec barre de progression
        for f in tqdm(downloaded_files, desc="Conversion FITS→PNG", ncols=80):
            try:
                fts_to_png(f, PNG_DIR)
                total_converted += 1
            except Exception as e:
                print(f"Erreur sur {f}: {e}")

        print(f"📦 Période {current.date()} → {window_end.date()} terminée ({total_converted} images traitées au total).")
        current = window_end

    print(f"\n🏁 Terminé : {total_converted} images converties et sauvegardées dans '{PNG_DIR}'.")

# =====================================================
# 3. POINT D'ENTRÉE
# =====================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Télécharge les images LASCO-C2 2022 et les convertit en PNG 512x512.")
    parser.add_argument("--start", type=str, default="2022-01-01", help="Date de début (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2022-12-31", help="Date de fin (YYYY-MM-DD)")
    args = parser.parse_args()

    start_time = datetime.fromisoformat(args.start)
    end_time = datetime.fromisoformat(args.end)

    download_and_convert_lasco_c2(start_time, end_time, step_days=30)
