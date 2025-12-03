import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from networks.models.segnet import CMEParamExtractor


# ===============================================================
# 1) Génération de masques polaires synthétiques (H=363, W=360)
# ===============================================================
def generate_polar_mask(pa_deg, aw_deg, H=363, W=360, thickness_ratio=0.4):
    """
    Génère un masque polaire artificiel conforme au format de la head.
    - pa_deg : position angulaire du centre (0°=droite, CCW)
    - aw_deg : angular width en degrés
    """

    # Cas particulier : pas de CME
    if aw_deg <= 0:
        return torch.zeros((1, 1, H, W), dtype=torch.float32)

    mask = torch.zeros((1, H, W), dtype=torch.float32)

    radial_thickness = int(H * thickness_ratio)
    radial_start = (H - radial_thickness) // 2
    radial_end = radial_start + radial_thickness

    half = aw_deg / 2
    amin = (pa_deg - half) % 360
    amax = (pa_deg + half) % 360

    angles = torch.linspace(0, 360, W+1)[:-1]  # W valeurs, 0 → 360 exclu

    if amin < amax:
        cond = (angles >= amin) & (angles <= amax)
    else:
        # wrap-around: intervalle qui traverse 360° → 0°
        cond = (angles >= amin) | (angles <= amax)

    mask[:, radial_start:radial_end, cond] = 1.0

    return mask.unsqueeze(0)  # shape (1, 1, H, W)

def add_noise_to_mask(mask, sigma=0.05):
    """
    Ajoute un bruit gaussien au masque (différentiable friendly).
    mask: (1, 1, H, W)
    sigma: écart-type du bruit (0.02–0.10 recommandé)

    Retourne :
        mask_bruite dans [0,1]
    """
    noise = torch.randn_like(mask) * sigma
    mask_noisy = mask + noise

    # Reforcer dans [0,1]
    mask_noisy = mask_noisy.clamp(0.0, 1.0)

    return mask_noisy



# ===============================================================
# 2) Fonction utilitaire pour afficher AW avec gestion wrap 360°
# ===============================================================
def plot_aw(ax, pa, aw, color, label, fill=True):
    """
    Affiche l'intervalle angular width (AW) en gérant correctement
    les cas wraps (chevauchement 360° -> 0°).
    """
    half = aw / 2
    a1 = (pa - half) % 360
    a2 = (pa + half) % 360

    # Cas simple : pas de wrap
    if a1 < a2:
        ax.axvspan(
            a1, a2,
            color=color if fill else "none",
            alpha=0.25 if fill else 1.0,
            edgecolor=color,
            linestyle="--" if not fill else "-",
            linewidth=2 if not fill else 1,
            label=label
        )
    else:
        # Cas wrap-around : [a1 → 360] et [0 → a2]
        ax.axvspan(
            a1, 360,
            color=color if fill else "none",
            alpha=0.25 if fill else 1.0,
            edgecolor=color,
            linestyle="--" if not fill else "-",
            linewidth=2 if not fill else 1,
            label=label
        )
        ax.axvspan(
            0, a2,
            color=color if fill else "none",
            alpha=0.25 if fill else 1.0,
            edgecolor=color,
            linestyle="--" if not fill else "-",
            linewidth=2 if not fill else 1
        )


# ===============================================================
# 3) Visualisation complète (option B) — Version corrigée
# ===============================================================
def visualize_prediction(mask, pa_true, aw_true, pa_pred, aw_pred):
    """
    Visualisation complète :
    - Masque polaire
    - Profil angulaire
    - Comparaison PA
    - Comparaison AW (avec wrap 360°)
    - Gestion correcte du cas AW=0 (pas de CME)
    """

    mask_np = mask.squeeze().cpu().numpy()
    H, W = mask_np.shape
    angles = np.linspace(0, 360, W, endpoint=False)
    density = mask_np.mean(axis=0)

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))

    # ---- (1) Masque polaire ----
    ax[0, 0].imshow(mask_np, cmap="gray", aspect="auto",
                    extent=[0, 360, 0, H])
    ax[0, 0].set_title("Polar Mask")
    ax[0, 0].set_xlabel("Angle (deg)")
    ax[0, 0].set_ylabel("Radius")

    # ---- (2) Profil angulaire ----
    ax[0, 1].plot(angles, density, label="Density")
    ax[0, 1].set_title("Angular Profile")
    ax[0, 1].set_xlabel("Angle (deg)")
    ax[0, 1].set_ylabel("Mean intensity")
    ax[0, 1].set_xlim(0, 360)

    # ---- (3) Position angulaire attendue vs prédite ----
    ax[1, 0].plot(angles, density)
    ax[1, 0].axvline(pa_true, color="green", linestyle="--", label="PA True")
    ax[1, 0].axvline(pa_pred, color="red", linestyle="-", label="PA Pred")
    ax[1, 0].set_title("Principal Angle Comparison")
    ax[1, 0].legend()

    # ---- (4) Angular Width Comparison ----
    ax[1, 1].set_title("Angular Width Comparison")

    # Recalcule les half-widths
    half_true = aw_true / 2
    half_pred = aw_pred / 2

    def aw_interval(pa, half):
        """Retourne (start,end) modulo 360°."""
        a1 = (pa - half) % 360
        a2 = (pa + half) % 360
        return a1, a2

    # ============================================================
    # AW TRUE (zone verte)
    # ============================================================
    if aw_true > 0:   # <--- correction clé
        a1_true, a2_true = aw_interval(pa_true, half_true)

        if a1_true < a2_true:
            ax[1,1].axvspan(a1_true, a2_true,
                            color="green", alpha=0.25,
                            label="AW True")
        else:
            ax[1,1].axvspan(a1_true, 360,
                            color="green", alpha=0.25,
                            label="AW True")
            ax[1,1].axvspan(0, a2_true,
                            color="green", alpha=0.25)

    # ============================================================
    # AW PRED (zone rouge hachurée)
    # ============================================================
    if aw_pred > 0:    # <--- correction identique
        a1_pred, a2_pred = aw_interval(pa_pred, half_pred)

        if a1_pred < a2_pred:
            ax[1,1].axvspan(a1_pred, a2_pred,
                            facecolor="none",
                            edgecolor="red",
                            hatch="//",
                            linewidth=1.5,
                            label="AW Pred")
        else:
            ax[1,1].axvspan(a1_pred, 360,
                            facecolor="none",
                            edgecolor="red",
                            hatch="//",
                            linewidth=1.5,
                            label="AW Pred")
            ax[1,1].axvspan(0, a2_pred,
                            facecolor="none",
                            edgecolor="red",
                            hatch="//",
                            linewidth=1.5)

    ax[1,1].set_xlim(0, 360)
    ax[1,1].legend()

    plt.tight_layout()
    plt.show()


# ===============================================================
# 4) Sanity Checker principal
# ===============================================================
class CMEHeadChecker:
    def __init__(self, H=363, W=360):
        self.H = H
        self.W = W
        self.head = CMEParamExtractor()

    # -------------------------------
    # Test d'un cas unique
    # -------------------------------
    def test(self, pa, aw, noise_sigma=0.0):
        """
        Test d’un cas unique, avec option d’ajout de bruit.
        """
        mask = generate_polar_mask(pa_deg=pa, aw_deg=aw,
                                H=self.H, W=self.W)

        # Appliquer le bruit si demandé
        if noise_sigma > 0:
            mask_eval = add_noise_to_mask(mask, sigma=noise_sigma)
        else:
            mask_eval = mask

        with torch.no_grad():
            out = self.head(mask_eval)
            has_cme, pa_pred, aw_pred = out[0]

        has_cme = float(has_cme)
        pa_pred = float(pa_pred)
        aw_pred = float(aw_pred)

        print("===== CME Head Test =====")
        print(f"PA True = {pa:6.2f}°   | PA Pred = {pa_pred:6.2f}°")
        print(f"AW True = {aw:6.2f}°   | AW Pred = {aw_pred:6.2f}°")
        print(f"has_cme_pred = {has_cme:.4f}")
        if noise_sigma > 0:
            print(f"(Noise applied: σ = {noise_sigma})")
        print()

        visualize_prediction(mask_eval.squeeze(0),
                            pa, aw,
                            pa_pred, aw_pred)


    # -------------------------------
    # Tests standards
    # -------------------------------
    def test_basic(self):
        tests = [
            (0, 30), (45, 40), (90, 60),
            (135, 30), (180, 50), (270, 20)
        ]
        for pa, aw in tests:
            self.test(pa, aw)
    
    def test_noisy(self):
        tests = [
            (90, 60)
        ]
        for pa, aw in tests:
            self.test(pa, aw, noise_sigma=0.05)

    def test_extremes(self):
        tests = [(0, 30), (90, 30), (180, 30), (270, 30)]
        for pa, aw in tests:
            self.test(pa, aw)

    def test_widths(self):
        tests = [(45, 10), (45, 30), (45, 60), (45, 120)]
        for pa, aw in tests:
            self.test(pa, aw)
