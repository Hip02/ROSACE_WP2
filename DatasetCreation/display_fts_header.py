#!/usr/bin/env python3
"""
Script pour afficher le header d’un fichier FITS/FITS LASCO.

Usage :
    python show_fits_header.py /chemin/vers/fichier.fts
"""

import sys
from astropy.io import fits
import os

def show_fits_header(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ Fichier introuvable : {filepath}")
    if not filepath.lower().endswith((".fits", ".fts")):
        raise ValueError("Le fichier doit être au format .fits ou .fts")

    print(f"🔍 Lecture du fichier FITS : {filepath}\n")

    # Ouvre le fichier FITS
    with fits.open(filepath) as hdul:
        print(f"Nombre d’extensions HDU : {len(hdul)}\n")

        # Extension principale (HDU 0)
        primary_header = hdul[0].header
        print("=== HEADER PRINCIPAL ===")
        for key, value in primary_header.items():
            print(f"{key:20} : {value}")

        # Si d’autres extensions existent (souvent vide pour LASCO)
        if len(hdul) > 1:
            for i, hdu in enumerate(hdul[1:], start=1):
                print(f"\n=== EXTENSION {i} ===")
                for key, value in hdu.header.items():
                    print(f"{key:20} : {value}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage : python show_fits_header.py fichier.fts")
        sys.exit(1)

    fits_file = sys.argv[1]
    show_fits_header(fits_file)
