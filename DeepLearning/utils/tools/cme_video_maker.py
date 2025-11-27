import os
import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from tqdm import tqdm


# -------------------------------------------------------------------------
# Reprend tes fonctions existantes
# -------------------------------------------------------------------------

def make_sector_mask_cartesian(H, W, pa_deg, aw_deg, inner_r=0, outer_r=None):
    if aw_deg <= 0:
        return torch.zeros((H, W))

    if outer_r is None:
        outer_r = min(H, W) // 2

    cx, cy = W // 2, H // 2
    Y, X = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")

    dx, dy = X - cx, Y - cy
    r = torch.sqrt(dx**2 + dy**2)
    angles = (torch.rad2deg(torch.atan2(dy, dx)) + 360) % 360

    amin = (pa_deg - aw_deg/2) % 360
    amax = (pa_deg + aw_deg/2) % 360

    if amin < amax:
        cond_a = (angles >= amin) & (angles <= amax)
    else:
        cond_a = (angles >= amin) | (angles <= amax)

    cond_r = (r >= inner_r) & (r <= outer_r)

    return (cond_a & cond_r).float()


def make_sector_mask_polar(Hp, Wp, pa_deg, aw_deg):
    if aw_deg <= 0:
        return torch.zeros((Hp, Wp))

    X = torch.linspace(0, 360, Wp)
    mask = torch.zeros((Hp, Wp))

    amin = (pa_deg - aw_deg/2) % 360
    amax = (pa_deg + aw_deg/2) % 360

    if amin < amax:
        mask[:, (X >= amin) & (X <= amax)] = 1
    else:
        mask[:, (X >= amin) | (X <= amax)] = 1

    return mask


# =========================================================================
#                     1 — CLASSE VIDÉO TEMPORALLE
# =========================================================================

class CMEVideoMaker:
    def __init__(self, DataLoaderClass, DatasetClass,
                 images_path, labels_path):

        loader = DataLoaderClass(
            root_dir=images_path,
            labels_file=labels_path,
            param=None
        )

        self.ds_cart = DatasetClass(
            dataloader=loader,
            param={"polar_transform": False, "shuffle": False}
        )

        self.ds_pol = DatasetClass(
            dataloader=loader,
            param={"polar_transform": True, "shuffle": False}
        )

        self.loader = loader


    def get_indices_for_month(self, year, month):
        idxs = []
        for i in range(len(self.ds_cart)):
            dt = self.ds_cart.get_datetime(i)
            if dt and dt.year == year and dt.month == month:
                idxs.append(i)
        return idxs


    def make_video(self, year, month, out_path="video.mp4",
                   fps=10, figsize=(10, 4)):

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

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        # ---- boucle temporelle ----
        for idx in tqdm(idxs, desc=f"Building CME video for {year}-{month:02d}", unit="frame"):
            fig = plt.figure(figsize=figsize)
            canvas = FigureCanvas(fig)

            ax1 = fig.add_subplot(1, 2, 1)
            ax2 = fig.add_subplot(1, 2, 2)

            # ---------------------------------------------------
            # load cartesian (NOUVEAU FORMAT)
            # ---------------------------------------------------
            sample_cart = self.ds_cart[idx]
            imgC = sample_cart["image"].squeeze().numpy()
            constraints = sample_cart["constraints"]  # LISTE dynamique
            dt = sample_cart["time"]
            dt_str = dt.strftime("%Y-%m-%d %H:%M")

            H, W = imgC.shape

            # ---------------------------------------------------
            # load polar
            # ---------------------------------------------------
            sample_pol = self.ds_pol[idx]
            imgP = sample_pol["image"].squeeze().numpy()
            Hp, Wp = imgP.shape

            # ---------------------------------------------------
            # CARTESIAN PANEL
            # ---------------------------------------------------
            ax1.imshow(imgC, cmap="gray")

            # Ajouter TOUS les CME simultanés
            for c in constraints:
                pa = c["pa"]
                da = c["da"]
                maskC = make_sector_mask_cartesian(H, W, pa, da).numpy()
                maskC_vis = np.ma.masked_where(maskC == 0, maskC)   # <— CLEAN
                ax1.imshow(maskC_vis, cmap="autumn", alpha=0.35)


            ax1.set_title(f"{dt_str}\n{len(constraints)} CME(s)", fontsize=10)
            ax1.axis("off")

            # ---------------------------------------------------
            # POLAR PANEL
            # ---------------------------------------------------
            ax2.imshow(imgP, cmap="gray", extent=[0, 360, 0, 6], aspect="auto")

            for c in constraints:
                pa = c["pa"]
                da = c["da"]
                maskP = make_sector_mask_polar(Hp, Wp, pa, da).numpy()
                maskP_vis = np.ma.masked_where(maskP == 0, maskP)
                ax2.imshow(maskP_vis, cmap="autumn", alpha=0.35,
                           extent=[0, 360, 0, 6], aspect="auto")

            ax2.set_title("Polaire", fontsize=10)
            ax2.set_xticks([0, 180, 360])
            ax2.set_yticks([0, 3, 6])

            fig.tight_layout()

            # Convert to image
            canvas.draw()
            buf = canvas.buffer_rgba()
            img = np.asarray(buf, dtype=np.uint8).reshape(h, w, 4)
            img = img[:, :, :3]

            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            video.write(img_bgr)

            plt.close(fig)

        video.release()
        print(f"🎥 Vidéo enregistrée : {out_path}")
