import os
import torch
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. GÉNÉRATION DE FAUX MASQUES
# ============================================================

def build_fake_mask(mask_type, H, W, device):
    """
    Génère divers masques pour tester la loss.
    """
    if mask_type == "constant":
        return torch.full((1,1,H,W), 0.5, device=device)

    elif mask_type == "random":
        return torch.rand((1,1,H,W), device=device)

    elif mask_type == "horizontal_stripe":
        m = np.zeros((H,W), dtype=np.float32)
        m[H//3:2*H//3, :] = 1.0
        return torch.tensor(m).unsqueeze(0).unsqueeze(0).to(device)

    elif mask_type == "vertical_stripe":
        m = np.zeros((H,W), dtype=np.float32)
        m[:, W//3:2*W//3] = 1.0
        return torch.tensor(m).unsqueeze(0).unsqueeze(0).to(device)

    elif mask_type == "gaussian_angle":
        theta = np.linspace(0, 2*np.pi, W)
        profile = np.exp(-0.5*((theta - np.pi) / 0.3)**2)
        m = np.tile(profile, (H,1))
        return torch.tensor(m).unsqueeze(0).unsqueeze(0).float().to(device)

    elif mask_type == "circular_edge":
        m = np.zeros((H,W), dtype=np.float32)
        m[:, -10:] = 1.0
        return torch.tensor(m).unsqueeze(0).unsqueeze(0).float().to(device)

    elif mask_type == "sawtooth_angle":
        saw = np.array([(i % 20) < 10 for i in range(W)], dtype=np.float32)
        m = np.tile(saw, (H,1))
        return torch.tensor(m).unsqueeze(0).unsqueeze(0).float().to(device)

    elif mask_type == "radial_step":
        m = np.zeros((H,W), dtype=np.float32)
        m[:H//2, :] = 1.0
        return torch.tensor(m).unsqueeze(0).unsqueeze(0).float().to(device)

    else:
        raise ValueError(f"Unknown mask_type={mask_type}")

# ============================================================
# 1.b CHARGEMENT DE MASQUES RÉELS
# ============================================================

def load_real_masks(network, max_masks=6):
    """
    Charge quelques masques prédits depuis :
        exp_list/<exp_name>/predicted_masks/*.pt
    Retourne une liste de tuples :
        (mask_tensor [1,1,R,W], constraints_batch)
    """

    pred_dir = os.path.join(network.results_path, "predicted_masks")
    if not os.path.exists(pred_dir):
        print("⚠ Aucun masque prédit trouvé. Lance d'abord test().")
        return []

    files = sorted([f for f in os.listdir(pred_dir) if f.endswith(".pt")])[:max_masks]
    real_masks = []

    for fname in files:
        data = torch.load(os.path.join(pred_dir, fname))
        mask = data["mask_pred"].unsqueeze(0).unsqueeze(0).to(network.device)   # [1,1,R,W]
        constraints = [data["constraints"]]

        real_masks.append((mask, constraints))

    return real_masks


# ============================================================
# 2. GÉNÉRATION DES ENSEMBLES DE CONTRAINTES
# ============================================================

def build_constraints_set(constraint_type):
    """
    Chaque contrainte suit ta structure :
    {theta_min, theta_max, pa, da}
    """
    if constraint_type == "noCME":
        return [[]]   # pour un batch de 1

    elif constraint_type == "CME_standard":
        return [[{
            "theta_min": 120,
            "theta_max": 180,
            "pa": 150,
            "da": 60,
        }]]

    elif constraint_type == "CME_wrap":
        return [[{
            "theta_min": 350,   # test wrap 350° → 10°
            "theta_max": 10,
            "pa": 0,
            "da": 20,
        }]]

    elif constraint_type == "CME_double":
        return [[
            {"theta_min": 80, "theta_max": 120, "pa": 100, "da": 40},
            {"theta_min": 200, "theta_max": 240, "pa": 220, "da": 40},
        ]]
    
    elif constraint_type == "CME_double2":
        return [[
            {"theta_min": 180, "theta_max": 220, "pa": 200, "da": 40},
            {"theta_min": 280, "theta_max": 320, "pa": 300, "da": 40},
        ]]

    elif constraint_type == "CME_double3":
        return [[
            {"theta_min": 20, "theta_max": 60, "pa": 40, "da": 40},
            {"theta_min": 180, "theta_max": 220, "pa": 200, "da": 40},
        ]]

    else:
        raise ValueError(f"Unknown constraint_type={constraint_type}")


# ============================================================
# 3. ANALYSE POUR UN MASQUE ET UNE CONTRAINTE
# ============================================================

def analyze_one(network, mask0, constraints_batch, output_dir, X):
    """
    Calcule le gradient de la loss pour un masque et une contrainte.
    Génère toutes les figures, y compris les deux profils angulaires superposés.
    """
    device = network.device
    os.makedirs(output_dir, exist_ok=True)

    mask_pred = mask0.clone().detach().requires_grad_(True)
    mask_np = mask0.detach().cpu().numpy()[0,0]

    # ======= FORWARD + BACKWARD =======
    loss, _ = network.criterion(mask_pred, constraints_batch, X)
    loss.backward()

    grad = mask_pred.grad.detach().cpu().numpy()[0,0]
    print(f"  Loss={loss.item():.6f}   grad=[{grad.min():.2e}, {grad.max():.2e}]")

    # ======= PROFIL ANGULAIRE PRED =======
    A_pred = mask_pred.detach().cpu().numpy()[0,0].mean(axis=0)  # [Theta]

    # ======= PROFIL ANGULAIRE TARGET =======
    # • Même structure que dans ta loss
    Theta = mask_pred.shape[-1]
    if len(constraints_batch[0]) > 0:
        T = torch.zeros(Theta, device=device)
        for c in constraints_batch[0]:
            tmin = int(c["theta_min"]) % Theta
            tmax = int(c["theta_max"]) % Theta
            if tmin <= tmax:
                T[tmin:tmax+1] = 1.0
            else:
                T[tmin:] = 1.0
                T[:tmax+1] = 1.0
        A_target = T.detach().cpu().numpy()
    else:
        A_target = np.zeros(Theta, dtype=np.float32)

    # ============================================================
    #   FIGURES EXISTANTES
    # ============================================================

    plt.figure(figsize=(6,6))
    plt.imshow(mask_np, cmap="gray")
    plt.title("Mask")
    plt.colorbar()
    plt.savefig(os.path.join(output_dir, "mask.png"))
    plt.close()

    plt.figure(figsize=(6,6))
    plt.imshow(grad, cmap="bwr")
    plt.title("Gradient")
    plt.colorbar()
    plt.savefig(os.path.join(output_dir, "grad_heatmap.png"))
    plt.close()

    plt.figure(figsize=(6,6))
    plt.imshow(np.abs(grad), cmap="viridis")
    plt.title("|Gradient|")
    plt.colorbar()
    plt.savefig(os.path.join(output_dir, "grad_norm.png"))
    plt.close()

    plt.figure(figsize=(6,4))
    plt.hist(grad.flatten(), bins=200)
    plt.title("Histogram")
    plt.savefig(os.path.join(output_dir, "grad_hist.png"))
    plt.close()

    # -------- Angular gradient profile --------
    ang_grad = grad.mean(axis=0)
    plt.figure(figsize=(6,4))
    plt.plot(ang_grad)
    plt.title("Angular Profile (Gradient)")
    plt.savefig(os.path.join(output_dir, "prof_angle.png"))
    plt.close()

    rad = grad.mean(axis=1)
    plt.figure(figsize=(6,4))
    plt.plot(rad)
    plt.title("Radial Profile")
    plt.savefig(os.path.join(output_dir, "prof_radial.png"))
    plt.close()

    fft = np.abs(np.fft.rfft(ang_grad))
    freqs = np.fft.rfftfreq(len(ang_grad))
    plt.figure(figsize=(6,4))
    plt.plot(freqs, fft)
    plt.title("FFT(gradient angle)")
    plt.savefig(os.path.join(output_dir, "fft.png"))
    plt.close()

    plt.figure(figsize=(6,6))
    plt.imshow(mask_np, cmap="gray")
    plt.imshow(grad, cmap="bwr", alpha=0.5)
    plt.title("Overlay")
    plt.savefig(os.path.join(output_dir, "overlay.png"))
    plt.close()

    # ============================================================
    # PLOT DES 2 PROFILS ANGULAIRES
    # ============================================================

    plt.figure(figsize=(7,4))
    plt.plot(A_pred, label="Pred (angular mean)", linewidth=2)
    plt.plot(A_target, label="Target (constraints)", linewidth=2)
    plt.title("Angular Profiles: pred vs target")
    plt.xlabel("Angle bin (0–359)")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, "prof_angle_pred_vs_target.png"))
    plt.close()


# ============================================================
# 4. SANITY CHECK COMPLET
# ============================================================

def sanity_check_gradient_analysis(network, exp_name="sanity_gradient", H=360, W=360, use_real_masks=False):

    base_path = os.path.join("exp_list", exp_name)
    os.makedirs(base_path, exist_ok=True)

    print("\n=== SANITY CHECK : GRADIENT ANALYSIS ===")

    # Placeholder X (structure compatible avec la loss)
    X = torch.zeros((1,5,H,W), device=network.device)

    # ------------------------------------------------------------------
    #  OPTION 1 : ANALYSE AVEC VRAIS MASQUES PRÉDITS
    # ------------------------------------------------------------------
    if use_real_masks:
        print("\n➡ MODE: REAL MASKS")

        real_masks = load_real_masks(network)

        if len(real_masks) == 0:
            print("❌ Aucun masque réel trouvé, fallback vers masques synthétiques.")
        else:
            out_dir = os.path.join(base_path, "real_masks")
            os.makedirs(out_dir, exist_ok=True)

            for k, (mask0, constraints_batch) in enumerate(real_masks):
                print(f"   → Real mask {k}")
                sample_dir = os.path.join(out_dir, f"sample_{k:03d}")
                os.makedirs(sample_dir, exist_ok=True)

                analyze_one(
                    network=network,
                    mask0=mask0,
                    constraints_batch=constraints_batch,
                    output_dir=sample_dir,
                    X=X,
                )

            print("\n✔ Sanity-check sur masques réels terminé.")
            return  # on s'arrête ici si use_real_masks=True

    # ------------------------------------------------------------------
    #  OPTION 2 : MODE SYNTHÉTIQUE (TA VERSION ACTUELLE)
    # ------------------------------------------------------------------
    print("\n➡ MODE: SYNTHETIC MASKS")

    mask_types = [
        "constant",
        "random",
        "horizontal_stripe",
        "vertical_stripe",
        "gaussian_angle",
        "circular_edge",
        "sawtooth_angle",
        "radial_step",
    ]

    constraint_types = [
        "noCME",
        "CME_standard",
        "CME_wrap",
        "CME_double",
        "CME_double2",
        "CME_double3"
    ]

    for mtype in mask_types:
        print(f"\n➡ MASK: {mtype}")
        mask_base_dir = os.path.join(base_path, f"mask_{mtype}")
        os.makedirs(mask_base_dir, exist_ok=True)

        mask0 = build_fake_mask(mtype, H, W, network.device)

        for ctype in constraint_types:
            print(f"   → Constraint: {ctype}")
            c_dir = os.path.join(mask_base_dir, f"{ctype}")
            os.makedirs(c_dir, exist_ok=True)

            constraints_batch = build_constraints_set(ctype)

            analyze_one(
                network=network,
                mask0=mask0,
                constraints_batch=constraints_batch,
                output_dir=c_dir,
                X=X,
            )

    print("\n✔ Sanity-check terminé : résultats dans", base_path)
