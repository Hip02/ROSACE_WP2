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

    def __init__(self, dataloader, param=None, seed=42):

        self.root_dir = dataloader.get_root_dir()
        self.labels_file = dataloader.get_labels_file()
        self.param = param or {}
        self.polar_transform = self.param.get("polar_transform", False)
        self.crop_polar_bottom = self.param.get("crop_polar_bottom", False)
        self.shuffle = self.param.get("shuffle", True)
        self.subsampling_factor = self.param.get("subsampling_factor", 1) if param else 1
        self.use_neighbor_diff = self.param.get("use_neighbor_diff", False)

        # -------------------------------------------------------
        # 1) Charger toutes les images et extraire leur datetime
        # -------------------------------------------------------
        self.image_paths = sorted([
            os.path.join(self.root_dir, f)
            for f in os.listdir(self.root_dir)
            if f.endswith(".png")
        ])

        if len(self.image_paths) == 0:
            raise ValueError(f"Aucune image PNG trouvée dans {self.root_dir}")

        random.seed(seed)
        np.random.seed(seed)

        self.cme_df = pd.read_csv(self.labels_file)
        self.cme_intervals = self._load_cme_intervals()
        self.labels_dict = self._create_labels_mapping(self.image_paths, self.cme_intervals)

        # --- extraction des timestamps pour chaque image ---
        self.image_times = [
            self._extract_datetime_from_filename(p)
            for p in self.image_paths
        ]

        # -------------------------------------------------------
        # 2) Construire indices triés par temps → pour adjacence t-1, t+1
        # -------------------------------------------------------
        self.sorted_indices_by_time = sorted(range(len(self.image_paths)),
                                             key=lambda i: self.image_times[i])

        # -------------------------------------------------------
        # 3) Shuffle & subsampling des indices
        # -------------------------------------------------------
        all_indices = list(range(len(self.image_paths)))

        if self.shuffle:
            random.shuffle(all_indices)

        self.data_indices = all_indices
        self.data_indices = self.data_indices[::self.subsampling_factor]


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
        """
        Retourne un dict:
            image_name -> [ {pa, da}, {pa, da}, ... ]  (liste de CME actives)
        La liste peut être de taille 0, 1, 2, 3, ...
        """
        labels_dict = {}

        for img_path in image_paths:
            img_name = os.path.basename(img_path)
            img_time = self._extract_datetime_from_filename(img_path)

            if img_time is None:
                labels_dict[img_name] = []
                continue

            # Liste dynamique, aucune limite sur le nombre
            active = []
            for cme in cme_intervals:
                if cme["start"] <= img_time <= cme["end"]:
                    active.append({
                        "pa_deg": cme["pa_deg"],
                        "da_deg": cme["da_deg"]
                    })

            labels_dict[img_name] = active

        return labels_dict


    def _build_angular_constraints(self, active_cmes):
        """
        Liste dynamique de contraintes, une par CME active.
        ex: [
            {"theta_min":..., "theta_max":..., "pa":..., "da":...},
            ...
        ]
        """
        constraints = []

        for cme in active_cmes:
            pa = (cme["pa_deg"] - 90) % 360
            da = cme["da_deg"]

            theta_min = (pa - da/2) % 360
            theta_max = (pa + da/2) % 360

            constraints.append({
                "theta_min": theta_min,
                "theta_max": theta_max,
                "pa": pa,
                "da": da
            })

        return constraints



    def get_label_infos_from_cme_id(self, cme_id):
        """
        Retourne un label au format (has_cme, pa_deg, da_deg) à partir du CME_ID.
        Si pas de CME, retourne (0.0, 0.0, 0.0).
        """
        if cme_id == 0:
            return torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)

        # Retrouver la CME correspondante
        match = next((cme for cme in self.cme_intervals if cme["id"] == cme_id), None)
        if match:
            pa_deg = match["pa_deg"]

            # |───────────────────────────────────────────────────────────────|
            # │ /!\  WARNING: Convention adjustment                           |
            # │ This converts CACTus position angle (PA) to the warp_polar    |
            # │ convention where 0° points to the right. Keep this shift.     |
            # │ to ensure angles align with the polar transform.              |
            # |───────────────────────────────────────────────────────────────|
            pa_deg = (pa_deg - 90) % 360


            da_deg = match["da_deg"]
            return torch.tensor([1.0, pa_deg, da_deg], dtype=torch.float32)
        else:
            return torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)


    def _apply_polar_transform(self, image_np, n_radial=360, n_theta=360):
        """
        Transforme une image 2D (numpy array) en coordonnées polaires.
        """
        # Calcul du centre
        center = (image_np.shape[0] / 2, image_np.shape[1] / 2)
        radius = np.hypot(center[0], center[1])

        # IMPORTANT : on impose explicitement le shape (r, theta)
        polar_image = warp_polar(
            image_np,
            center=center,
            radius=radius,
            scaling='linear',
            output_shape=(n_radial, n_theta)
        )

        # On permute pour avoir (theta, r)
        polar_image = np.transpose(polar_image)   # shape → [n_theta, n_radial]
        polar_image = np.flipud(polar_image)

        # Normalisation [0,1]
        polar_image = np.clip(polar_image, 0.0, 1.0)
        return polar_image



    def __len__(self):
        return len(self.data_indices)

    def get_datetime(self, idx):
        """
        Retourne la datetime exacte correspondant au sample idx,
        en prenant en compte le split (train/val/test).
        """
        real_idx = self.data_indices[idx]
        img_path = self.image_paths[real_idx]
        return self._extract_datetime_from_filename(img_path)

    def _load_polar_tensor(self, path):
        image = Image.open(path).convert("L")
        image_np = np.array(image, dtype=np.float32) / 255.0

        # 1) Polar transform (optional)
        if self.polar_transform:
            image_np = self._apply_polar_transform(image_np)
        
         # 2) Radial cropping (optional)
        if self.crop_polar_bottom:
            # crop_polar_bottom = fraction à retirer (ex: 0.22)
            frac = 0.22
            H = image_np.shape[0]
            keep = int(H * (1.0 - frac))

            # On garde uniquement les lignes du haut
            image_np = image_np[:keep, :]

        return torch.tensor(image_np, dtype=torch.float32).unsqueeze(0)   # [1,H,W]

    # =====================================================================
    #                               GETITEM
    # =====================================================================
    def __getitem__(self, idx):
        # real index dans image_paths
        real_idx = self.data_indices[idx]
        img_path = self.image_paths[real_idx]

        # Charger image t
        I_t = self._load_polar_tensor(img_path)

        # -----------------------------------------------------
        # Trouver t-1 et t+1 dans **la liste triée par temps**
        # -----------------------------------------------------
        sorted_pos = self.sorted_indices_by_time.index(real_idx)

        # voisin t-1
        if sorted_pos > 0:
            idx_prev = self.sorted_indices_by_time[sorted_pos - 1]
            I_prev = self._load_polar_tensor(self.image_paths[idx_prev])
        else:
            I_prev = torch.zeros_like(I_t)

        # voisin t+1
        if sorted_pos < len(self.sorted_indices_by_time) - 1:
            idx_next = self.sorted_indices_by_time[sorted_pos + 1]
            I_next = self._load_polar_tensor(self.image_paths[idx_next])
        else:
            I_next = torch.zeros_like(I_t)

        # -----------------------------------------------------
        # Images différentielles
        # -----------------------------------------------------
        dI_prev = I_t - I_prev
        dI_next = I_next - I_t

        if self.use_neighbor_diff:
            # Stack final : [5,H,W]
            img_stack = torch.cat([I_prev, I_t, I_next, dI_prev, dI_next], dim=0)
        else:
            # Stack final : [1,H,W]
            img_stack = I_t

        # -----------------------------------------------------
        # Labels / constraints
        # -----------------------------------------------------
        img_name = os.path.basename(img_path)
        active_cmes = self.labels_dict.get(img_name, [])
        constraints = self._build_angular_constraints(active_cmes)

        t_frame = self._extract_datetime_from_filename(img_path)

        return {
            "image": img_stack,            # [5, H, W] ou [1, H, W] si pas de voisins
            "constraints": constraints,  # liste dynamique selon CME actives
            "time": t_frame
        }

