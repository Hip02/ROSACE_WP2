import torch
import matplotlib.pyplot as plt
import random


# -------------------------------------------------------------------------
# --- 1. Masques synthétiques
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


# -------------------------------------------------------------------------
# --- 2. Sélection ≥ 3 CME positives
# -------------------------------------------------------------------------

def find_examples(dataset, n=5, min_cme=3):
    """
    Sélectionne au moins min_cme exemples positifs et complète avec
    des négatifs, mais en mélangeant au préalable les indices
    pour éviter les séquences consécutives.
    """
    indices = list(range(len(dataset)))

    # Random object indépendant
    rd_obj = random.Random()

    rd_obj.shuffle(indices)   # ← clé pour éviter les samples consécutifs !

    positives = []
    negatives = []

    for i in indices:
        _, label = dataset[i]
        has_cme = label[0]

        if has_cme == 1 and len(positives) < min_cme:
            positives.append(i)
        elif has_cme == 0:
            negatives.append(i)

        if len(positives) >= min_cme and len(positives)+len(negatives) >= n:
            break

    # Combine et re-mélange légèrement la sélection finale
    selected = positives + negatives[:(n - len(positives))]
    rd_obj.shuffle(selected)

    return selected


# -------------------------------------------------------------------------
# --- 3. Classe principale compatible avec ton format (image, label)
# -------------------------------------------------------------------------

class PolarTransformChecker:
    def __init__(self, DataLoaderClass, DatasetClass,
                 images_path, labels_path,
                 n_examples=5, min_cme=3):

        self.DataLoaderClass = DataLoaderClass
        self.DatasetClass = DatasetClass
        self.images_path = images_path
        self.labels_path = labels_path
        self.n_examples = n_examples
        self.min_cme = min_cme

        # --- Loader global ---
        loader = DataLoaderClass(
            root_dir=images_path,
            labels_file=labels_path,
            param=None
        )

        # --- Dataset cartésien ---
        self.ds_cart = DatasetClass(
            dataloader=loader,
            mode="train",
            param={"polar_transform": False, "shuffle": False}
        )

        # --- Sélection d'exemples ---
        self.ids = find_examples(self.ds_cart, n_examples, min_cme)

        # --- Dataset polaire ---
        self.ds_pol = DatasetClass(
            dataloader=loader,
            mode="train",
            param={"polar_transform": True, "shuffle": False}
        )

    # ---------------------------------------------------------------------
    def check(self, n=6):
        # --- Sélection d'exemples (6) ---
        self.ids = find_examples(self.ds_cart, n=n, min_cme=self.min_cme)

        # --- Figure : 3 lignes × 4 colonnes ---
        fig, axes = plt.subplots(3, 4, figsize=(14, 9))
        fig.suptitle("Sanity Check – Cartesian vs Polar Transform",
                    fontsize=16, fontweight="bold", y=0.995)

        # Couleurs d'encadrement
        CME_COLOR = (0.0, 0.9, 0.0)   # vert pour CME=1
        NO_CME_COLOR = (1.0, 1.0, 1.0)  # blanc pour CME=0

        for i, idx in enumerate(self.ids):
            row = i // 2
            col_base = (i % 2) * 2  # 0 ou 2

            # ---------------------------------------------------
            #                 LOAD DATA + METADATA
            # ---------------------------------------------------
            imgC, labelC = self.ds_cart[idx]
            imgC = imgC.squeeze()

            has_cme = int(labelC[0].item())
            pa = labelC[1].item()
            aw = labelC[2].item()

            dt = self.ds_cart.get_datetime(idx)
            dt_str = dt.strftime("%Y-%m-%d %H:%M") if dt else "???"

            border_color = CME_COLOR if has_cme == 1 else NO_CME_COLOR

            # Masks
            H, W = imgC.shape
            maskC = make_sector_mask_cartesian(H, W, pa, aw)

            imgP, _ = self.ds_pol[idx]
            imgP = imgP.squeeze()
            Hp, Wp = imgP.shape
            maskP = make_sector_mask_polar(Hp, Wp, pa, aw)

            # ---------------------------------------------------
            #                     CARTESIAN
            # ---------------------------------------------------
            axC = axes[row, col_base]
            axC.imshow(imgC, cmap="gray")
            axC.imshow(maskC, cmap="autumn", alpha=0.35)
            axC.set_title(f"{dt_str}\nCME={has_cme}", fontsize=9, pad=4)
            axC.set_xticks([])
            axC.set_yticks([])

            axC.add_patch(plt.Rectangle(
                (0, 0), W, H, fill=False,
                edgecolor=border_color, linewidth=6.0
            ))

            # ---------------------------------------------------
            #                       POLAR
            # ---------------------------------------------------
            axP = axes[row, col_base + 1]
            axP.imshow(imgP, cmap="gray", extent=[0, 360, 0, 6], aspect="auto")
            axP.imshow(maskP, cmap="autumn", alpha=0.35,
                    extent=[0, 360, 0, 6], aspect="auto")

            axP.set_title("Polar", fontsize=9, pad=4)
            axP.set_xticks([0, 180, 360])
            axP.set_yticks([0, 3, 6])

            axP.add_patch(plt.Rectangle(
                (0, 0), 360, 6, fill=False,
                edgecolor=border_color, linewidth=6.0
            ))

            # Style général propre
            axP.tick_params(axis='both', which='major', labelsize=8)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
