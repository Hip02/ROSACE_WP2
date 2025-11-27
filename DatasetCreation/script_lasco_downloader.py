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
import os
from datetime import datetime
import re

import warnings

# Ignore les warnings LASCO typiques
warnings.filterwarnings("ignore", message="Missing metadata for solar radius")
warnings.filterwarnings("ignore", message="Missing metadata for observer")
warnings.filterwarnings("ignore", message="Assuming Earth-based observer")

import logging

# Désactive tous les logs SunPy
logging.getLogger('sunpy').setLevel(logging.ERROR)
logging.getLogger('sunpy.map').setLevel(logging.ERROR)
logging.getLogger('sunpy.map.mapbase').setLevel(logging.ERROR)


import sunpy.map
from sunpy.map import Map

# =====================================================
# 1.5. FONCTION DE CONVERSION .fts → .png
# =====================================================

def fts_to_png(input_path, output_dir, size=(512, 512), contrast_clip=(1, 99)):
    """
    Convertit un fichier LASCO .fts/.fits en PNG 512x512.
    - Ignore (skip) les images polarisées
    - Le nom du PNG = basé sur DATE-OBS + TIME-OBS (comme avant)
    """

    try:
        # Lecture SunPy
        smap = Map(input_path)
        header = smap.meta

        # 🔥 Filtrage des images polarisées (skip, mais ne supprime rien)
        polar_field = str(header.get("POLAR", "")).lower()

        # Tous les cas de polarisation LASCO C2 : "-60 deg", "120 deg", "seq pw", etc.
        if ("deg" in polar_field) or ("pw" in polar_field) or ("pol" in polar_field):
            #print(f"⏭️  Image polarisée ignorée : {input_path}")
            return None

        # ============================
        # Rotation WCS correcte
        # ============================
        try:
            smap = smap.rotate(order=3, missing=0, recenter=True)
        except Exception:
            print("⚠️ Rotation impossible → version brute utilisée")
        
        data = smap.data.astype(float)

        # ============================
        # Correction photométrique
        # ============================
        exptime = header.get("EXPTIME", None)
        if exptime is not None and exptime > 0:
            data = data / exptime

        # ============================
        # Nettoyage numérique
        # ============================
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        # ============================
        # Lecture de la date
        # ============================
        date_part = header.get("DATE-OBS", "")
        time_part = header.get("TIME-OBS", "")

        if date_part and time_part:
            date_full = f"{date_part}T{time_part}"
        elif date_part:
            date_full = date_part
        else:
            date_full = os.path.splitext(os.path.basename(input_path))[0]

        # Parsing robuste
        date_obj = None
        for fmt in ("%Y/%m/%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
            try:
                date_obj = datetime.strptime(date_full, fmt)
                break
            except Exception:
                pass

        if date_obj:
            date_str = date_obj.strftime("%Y%m%d_%H%M%S")
        else:
            # fallback simple
            date_str = re.sub(r'[^0-9]', '_', date_full)

        # ============================
        # Normalisation dynamique
        # ============================
        vmin, vmax = np.percentile(data, contrast_clip)
        data = np.clip(data, vmin, vmax)
        data = (255 * (data - vmin) / (vmax - vmin)).astype(np.uint8)

        # ============================
        # Resize
        # ============================
        img = Image.fromarray(data)
        img = img.resize(size, Image.Resampling.LANCZOS)

        # ============================
        # Nom final → basé sur la date UNIQUEMENT
        # ============================
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

def download_and_convert_lasco_c2(
    start_time=None,
    end_time=None,
    step_days=30,
    local_only=False,
    fts_dir=DATA_DIR,
    png_dir=PNG_DIR
):
    """
    Télécharge et/ou convertit des images LASCO-C2.
    
    Modes :
      - local_only=False : télécharge entre start_time et end_time puis convertit.
      - local_only=True  : ignore le téléchargement, liste fts_dir et convertit tous les fichiers .fts/.fits.

    Paramètres :
        start_time (datetime)
        end_time   (datetime)
        step_days  : taille des fenêtres temporelles pour le téléchargement
        local_only : si True, ne télécharge rien
        fts_dir    : dossier contenant les fichiers .fts déjà téléchargés
        png_dir    : dossier de sortie des PNG
    """

    total_converted = 0

    # ============================================================
    # MODE 1 : Conversion uniquement des fichiers locaux
    # ============================================================
    if local_only:
        print(f"\n📁 Mode local activé : conversion des fichiers dans {fts_dir}\n")

        # Lister les fichiers .fts/.fits du dossier
        all_files = []
        for root, dirs, files in os.walk(fts_dir):
            for f in files:
                if f.lower().endswith((".fts", ".fits")):
                    all_files.append(os.path.join(root, f))

        if len(all_files) == 0:
            print("⚠️ Aucun fichier FTS trouvé dans ce dossier.")
            return

        print(f"🔍 {len(all_files)} fichiers trouvés. Conversion en cours...\n")

        for f in tqdm(all_files, desc="Conversion FITS→PNG", ncols=80):
            try:
                fts_to_png(f, png_dir)
                total_converted += 1
            except Exception as e:
                print(f"❌ Erreur sur {f} : {e}")

        print(f"\n🏁 Conversion locale terminée : {total_converted} images PNG créées.")
        return


    # ============================================================
    # MODE 2 : Télécharger + convertit
    # ============================================================
    current = start_time
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

        # Téléchargement
        downloaded_files = []
        for i in tqdm(range(n_results), desc="Téléchargement FITS", ncols=80):
            try:
                file_list = Fido.fetch(
                    query[0][i:i+1],
                    progress=False,
                    path=os.path.join(fts_dir, "{file}")
                )
                if file_list:
                    downloaded_files.append(file_list[0])
            except Exception as e:
                print(f"Erreur sur téléchargement {i}: {e}")

        print(f"✅ {len(downloaded_files)} fichiers téléchargés. Conversion en PNG...")

        # Conversion
        for f in tqdm(downloaded_files, desc="Conversion FITS→PNG", ncols=80):
            try:
                fts_to_png(f, png_dir)
                total_converted += 1
            except Exception as e:
                print(f"Erreur sur {f}: {e}")

        print(f"📦 Période {current.date()} → {window_end.date()} terminée ({total_converted} images traitées au total).")
        current = window_end

    print(f"\n🏁 Terminé : {total_converted} images converties et sauvegardées dans '{png_dir}'.")


# =====================================================
# 3. POINT D'ENTRÉE
# =====================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Télécharge ou convertit localement les images LASCO-C2 en PNG 512x512."
    )

    # Dates (utilisées uniquement si local_only=False)
    parser.add_argument("--start", type=str, default="2022-01-01",
                        help="Date de début (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2022-12-31",
                        help="Date de fin (YYYY-MM-DD)")

    # Nouveau paramètre : conversion locale uniquement
    parser.add_argument("--local-only", action="store_true",
                        help="Ne télécharge rien, convertit uniquement les fichiers .fts déjà présents dans DATA_DIR.")

    # Dossiers optionnels (par défaut DATA_DIR et PNG_DIR globales)
    parser.add_argument("--fts-dir", type=str, default=DATA_DIR,
                        help="Dossier contenant les fichiers FTS locaux.")
    parser.add_argument("--png-dir", type=str, default=PNG_DIR,
                        help="Dossier où sauvegarder les PNG.")

    args = parser.parse_args()

    # Conversion des dates uniquement si on est en mode téléchargement
    if not args.local_only:
        start_time = datetime.fromisoformat(args.start)
        end_time = datetime.fromisoformat(args.end)
    else:
        start_time = None
        end_time = None

    # Appel principal
    download_and_convert_lasco_c2(
        start_time=start_time,
        end_time=end_time,
        step_days=30,
        local_only=args.local_only,
        fts_dir=args.fts_dir,
        png_dir=args.png_dir
    )
