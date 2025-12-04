import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from tqdm import tqdm


# =========================================================================
# 0 — POLAR → CARTESIAN (générique, propre, indépendant de ton dataset)
# =========================================================================

def polar_to_cartesian_mask(mask_polar, H, W):
    """
    mask_polar : [Hp, Wp] numpy or torch
    Retourne un masque cartésien [H, W] numpy.
    """

    if isinstance(mask_polar, torch.Tensor):
        mask_polar = mask_polar.cpu().numpy()

    Hp, Wp = mask_polar.shape

    # --- Correction : l'image polaire de LASCO a l'axe radial inversé ---
    mask_polar = np.flipud(mask_polar)

    # Grille cartésienne normale
    cx, cy = W//2, H//2
    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

    # Coordonnées polaires
    dx = x - cx
    dy = y - cy

    r = np.sqrt(dx**2 + dy**2)
    theta = (np.degrees(np.arctan2(dy, dx)) + 360) % 360

    # Normalisation pour indexer le masque polaire
    theta_norm = theta / 360 * (Wp - 1)
    r_norm = r / r.max() * (Hp - 1)

    # Interpolation bilinéaire
    x0 = np.floor(theta_norm).astype(int)
    x1 = (x0 + 1) % Wp
    y0 = np.floor(r_norm).astype(int)
    y1 = np.clip(y0 + 1, 0, Hp - 1)

    wa = (theta_norm - x0)
    wb = (r_norm - y0)

    # 4 contributions
    Ia = mask_polar[y0, x0]
    Ib = mask_polar[y0, x1]
    Ic = mask_polar[y1, x0]
    Id = mask_polar[y1, x1]

    w00 = (1 - wa) * (1 - wb)
    w01 = wa * (1 - wb)
    w10 = (1 - wa) * wb
    w11 = wa * wb

    return Ia*w00 + Ib*w01 + Ic*w10 + Id*w11



# =========================================================================
# 1 — CLASSE VIDÉO AVEC PREDICTIONS DU MODÈLE
# =========================================================================

class CMEVideoPredMaker:
    def __init__(self,
                 dataLoader,            # ← utilisé pour le modèle
                 DatasetClass,          # ← utilisé pour la vidéo
                 network,               # model déjà construit et chargé
                 device="cpu"):

        self.device = device
        self.model = network.to(device)
        self.model.eval()

        base_params = {}
        if hasattr(dataLoader, "get_param"):
            base_params = dataLoader.get_param()
        elif hasattr(dataLoader, "param"):
            base_params = dataLoader.param or {}

        # Dataset cartésien pour la vidéo
        self.ds_cart = DatasetClass(
            dataloader=dataLoader,
            param={**base_params, "polar_transform": False, "shuffle": False}
        )

        # Dataset polaire pour la vidéo
        self.ds_pol = DatasetClass(
            dataloader=dataLoader,
            param={**base_params, "polar_transform": True, "shuffle": False}
        )

        # IMPORTANT : garder dataLoader tel quel pour les mêmes stacks
        self.loader = dataLoader


    # ----------------------------------------------------------------------
    # Obtenir les indices d'un mois donné
    # ----------------------------------------------------------------------
    def get_indices_for_month(self, year, month):
        idxs = []
        for i in range(len(self.ds_cart)):
            dt = self.ds_cart.get_datetime(i)
            if dt and dt.year == year and dt.month == month:
                idxs.append(i)
        return idxs


    # ----------------------------------------------------------------------
    # Vidéo avec masques prédits
    # ----------------------------------------------------------------------
    def make_video(self, year, month,
               out_path="video_pred.mp4",
               fps=10, figsize=(10, 4),
               show_background=True):


        idxs = self.get_indices_for_month(year, month)
        if len(idxs) == 0:
            raise ValueError(f"Aucune image trouvée pour {year}-{month:02d}")

        print(f"→ {len(idxs)} images trouvées pour {year}-{month:02d}")

        tmp_fig = plt.figure(figsize=figsize)
        canvas = FigureCanvas(tmp_fig)
        tmp_fig.tight_layout()
        canvas.draw()
        w, h = canvas.get_width_height()
        plt.close(tmp_fig)

        video = cv2.VideoWriter(
            out_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h)
        )

        for idx in tqdm(idxs, desc="Building prediction video", unit="frame"):

            # -------------- CARTESIAN FRAME --------------
            sc = self.ds_cart[idx]
            center_idx = self.ds_cart.get_center_channel_index()
            imgC = sc["image"][center_idx].numpy()
            dt = sc["time"]
            dt_str = dt.strftime("%Y-%m-%d %H:%M")
            H, W = imgC.shape

            # -------------- POLAR STACK ------------------
            sp = self.ds_pol[idx]
            imgP_stack = sp["image"].to(self.device)  # <= [C,Hp,Wp]
            Hp, Wp = imgP_stack.shape[-2:]

            # ---------- PREDICTION MODEL ----------
            with torch.no_grad():
                pred = self.model(imgP_stack.unsqueeze(0))  # [1,C,H,W] -> [1,1,H,W]
                pred = torch.sigmoid(pred)
                maskP = pred.squeeze().cpu().numpy()

            # --- polar → cartesian
            maskC = polar_to_cartesian_mask(maskP, H, W)

            # Visualization masks
            maskP_vis = np.ma.masked_where(maskP < 0.3, maskP)
            maskC_vis = np.ma.masked_where(maskC < 0.3, maskC)

            # ------------ FIGURE -------------
            fig = plt.figure(figsize=figsize)
            canvas = FigureCanvas(fig)
            ax1 = fig.add_subplot(1, 2, 1)
            ax2 = fig.add_subplot(1, 2, 2)

            if show_background:
                ax1.imshow(imgC, cmap="gray")
                ax1.imshow(maskC_vis, cmap="autumn", alpha=0.40)
            else:
                ax1.imshow(maskC_vis, cmap="autumn", alpha=1.0)

            ax1.set_title(f"{dt_str}\nPred mask", fontsize=10)
            ax1.axis("off")

            imgP_vis = imgP_stack[self.ds_pol.get_center_channel_index()].cpu().numpy()
            ax2.imshow(imgP_vis, cmap="gray",
                       extent=[0, 360, 0, 6], aspect="auto")
            ax2.imshow(maskP_vis, cmap="autumn", alpha=0.40,
                       extent=[0, 360, 0, 6], aspect="auto")
            ax2.set_title("Pred polar", fontsize=10)
            ax2.set_xticks([0, 180, 360])
            ax2.set_yticks([0, 3, 6])

            fig.tight_layout()

            canvas.draw()
            buf = canvas.buffer_rgba()
            img = np.asarray(buf, dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
            video.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            plt.close(fig)

        video.release()
        print(f"🎥 Vidéo enregistrée : {out_path}")
