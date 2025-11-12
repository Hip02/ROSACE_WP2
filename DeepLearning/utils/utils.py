import os
import random
from PIL import Image
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from skimage.transform import warp_polar
from datetime import datetime, timedelta


class MyDataLoader:
    def __init__(self, root_dir, labels_file, param=None):
        self.root_dir = root_dir
        self.labels_file = labels_file
        self.param = param

    def get_root_dir(self):
        return self.root_dir
    
    def get_labels_file(self):
        return self.labels_file
        

class LascoC2ImagesDataset(Dataset):
    """
    Dataset pour charger des images PNG de LASCO C2.
    Divise le dataset en train / val / test de manière aléatoire mais reproductible.
    """

    def __init__(self, dataloader, mode="train", param=None, split_ratio=(0.8, 0.1, 0.1), seed=42):
        assert mode in ["train", "val", "test"], "mode doit être 'train', 'val' ou 'test'"
        self.root_dir = dataloader.get_root_dir()
        self.labels_file = dataloader.get_labels_file()
        self.mode = mode
        self.param = param

        self.polar_transform = param.get("polar_transform", False) if param else False

        # Trouver toutes les images PNG
        self.image_paths = sorted([
            os.path.join(self.root_dir, f)
            for f in os.listdir(self.root_dir)
            if f.endswith(".png")
        ])
        if len(self.image_paths) == 0:
            raise ValueError(f"Aucune image PNG trouvée dans {self.root_dir}")

        # Reproductibilité pour le split
        random.seed(seed)
        np.random.seed(seed)

        # Charger les infos CME depuis le CSV
        self.cme_df = pd.read_csv(self.labels_file)
        self.cme_intervals = self._load_cme_intervals()
        self.labels_dict = self._create_labels_mapping(self.image_paths, self.cme_intervals)

        # Split train/val/test
        indices = list(range(len(self.image_paths)))
        random.shuffle(indices)

        n_total = len(indices)
        n_train = int(split_ratio[0] * n_total)
        n_val = int(split_ratio[1] * n_total)

        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]

        if mode == "train":
            self.data_indices = train_indices
        elif mode == "val":
            self.data_indices = val_indices
        else:
            self.data_indices = test_indices


    def _load_cme_intervals(self):
        """Lit le CSV et crée une liste d'intervalles temporels avec un ID unique."""
        cme_intervals = []
        for idx, row in self.cme_df.iterrows():
            try:
                start = datetime.strptime(row["t0"], "%Y-%m-%d %H:%M:%S")
                duration = float(row["dt0_h"])
                end = start + timedelta(hours=duration)

                cme_intervals.append({
                    "id": f"CME_{idx}",  # ✅ ID unique basé sur l'index du CSV
                    "start": start,
                    "end": end,
                    "pa_deg": float(row["pa_deg"]),
                    "da_deg": float(row["da_deg"])
                })
            except Exception as e:
                print(f"⚠️ Erreur parsing ligne {idx}: {e}")

        return cme_intervals


    def _extract_datetime_from_filename(self, filename):
        """Extrait la date/heure du nom du fichier image LASCO."""
        try:
            parts = os.path.basename(filename).split("_")
            date_str = f"{parts[1]}-{parts[2]}-{parts[3]} {parts[4]}:{parts[5]}:{parts[6].split('.')[0]}"
            return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
        except Exception as e:
            print(f"⚠️ Erreur parsing date dans {filename}: {e}")
            return None


    def _create_labels_mapping(self, image_paths, cme_intervals):
        """Crée un dictionnaire image_name → CME_ID (ou 0 si aucune CME)."""
        labels_dict = {}

        for img_path in image_paths:
            img_time = self._extract_datetime_from_filename(img_path)
            if img_time is None:
                labels_dict[os.path.basename(img_path)] = 0
                continue

            cme_id = 0
            for cme in cme_intervals:
                if cme["start"] <= img_time <= cme["end"]:
                    cme_id = cme["id"]
                    break

            labels_dict[os.path.basename(img_path)] = cme_id

        return labels_dict


    def get_label_infos_from_cme_id(self, cme_id):
        """
        Retourne un label au format (has_cme, pa_deg, da_deg) à partir du CME_ID.
        Si pas de CME, retourne (0.0, -1.0, -1.0).
        """
        if cme_id == 0:
            return torch.tensor([0.0, -1.0, -1.0], dtype=torch.float32)

        # Retrouver la CME correspondante
        match = next((cme for cme in self.cme_intervals if cme["id"] == cme_id), None)
        if match:
            pa_deg = match["pa_deg"]
            da_deg = match["da_deg"]
            return torch.tensor([1.0, pa_deg, da_deg], dtype=torch.float32)
        else:
            return torch.tensor([0.0, -1.0, -1.0], dtype=torch.float32)


    def _apply_polar_transform(self, image_np):
        """
        Transforme une image 2D (numpy array) en coordonnées polaires.
        - r : distance radiale depuis le centre
        - theta : angle azimutal (0–360°)
        """
        # Calcul du centre de l’image
        center = (image_np.shape[0] / 2, image_np.shape[1] / 2)
        # Rayon maximal (jusqu’aux coins)
        radius = np.hypot(center[0], center[1])

        # Transformation cartésienne → polaire
        polar_image = warp_polar(
            image_np,
            center=center,
            radius=radius,
            scaling='linear',   # conserve les distances radiales réelles
            output_shape=None   # auto
        )

        # On permute les axes pour avoir l'angle sur l'axe X
        polar_image = np.transpose(polar_image)  # (r, θ) -> (θ, r)
        polar_image = np.flipud(polar_image)      # inverse l'axe vertical (r croît vers le bas)

        # Normalisation entre 0 et 1
        polar_image = np.clip(polar_image, 0.0, 1.0)

        return polar_image


    def __len__(self):
        return len(self.data_indices)


    def __getitem__(self, idx):
        real_idx = self.data_indices[idx]
        img_path = self.image_paths[real_idx]

        # Charger l'image en niveau de gris
        image = Image.open(img_path).convert("L")
        image_np = np.array(image, dtype=np.float32) / 255.0

        # Si la transformation polaire est activée
        if self.polar_transform:
            image_np = self._apply_polar_transform(image_np)

        # Conversion en tensor [1, H, W]
        image = torch.tensor(image_np, dtype=torch.float32).unsqueeze(0)

        # Récupérer le label
        img_name = os.path.basename(img_path)
        cme_id = self.labels_dict.get(img_name, 0)
        label = self.get_label_infos_from_cme_id(cme_id)

        return image, label