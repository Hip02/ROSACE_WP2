import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.utils import LascoC2ImagesDataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import random
import math
from datetime import datetime
from networks.architectures.network_1 import CMESegNet, WeakCMECompositeLoss, WeakCMEUNet, CMELossAngularProfileMSE, CMELossAngularProfileMSE_V2, CMELossAngularProfileMSE_V3, CMELossAngularProfileMSE_V4

# ========================
# 1. UTILITAIRES
# ========================
def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)

def create_folder(path: str):
    if not os.path.exists(path):
        os.makedirs(path)

def weak_collate(batch):
    """
    batch = list of samples:
        {
            "image": Tensor[1,H,W],
            "constraints": list[dict],
            "time": datetime
        }
    """
    images = torch.stack([item["image"] for item in batch], dim=0)
    constraints = [item["constraints"] for item in batch]  # PAS empilé
    times = [item["time"] for item in batch]

    return {
        "image": images,           # Tensor[B,1,H,W]
        "constraints": constraints, # list-of-lists, intact
        "time": times
    }


# ========================
# 2. CLASSE ENTRAÎNEMENT/TEST
# ========================
class Network:
    def __init__(self, dataLoader, param={}, exp_name="Unnamed", device="cpu"):
        set_seed(42)
        self.dataLoader = dataLoader
        self.device = device
        self.results_path = "results/exps/" + exp_name
        self.epochs = param.get("epochs", 5)
        self.batchSize = param.get("batch_size", 16)
        self.lr = param.get("lr", 1e-3)
        self.use_neighbors_diff = param.get("use_neighbor_diff", False) if param is not None else False
        
        create_folder(self.results_path)

        # Data Loader
        self.dataSet = LascoC2ImagesDataset(self.dataLoader, param=param)

        self.dataLoader = DataLoader(self.dataSet, batch_size=self.batchSize, shuffle=True, collate_fn=weak_collate, num_workers=0)

        # Model + Optimizer + Loss
        self.in_channels = 5 if self.use_neighbors_diff else 1

        self.model = WeakCMEUNet(in_channels=self.in_channels).to(device)
        self.criterion = CMELossAngularProfileMSE_V4().to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train(self):
        train_losses = []

        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0

            for batch in tqdm(self.dataLoader, desc=f"Epoch {epoch+1}/{self.epochs}"):

                # Batch extraction
                X = batch["image"].to(self.device)             # [B,1,R,Theta]
                constraints_batch = batch["constraints"]       # list of lists (CPU OK)

                self.optimizer.zero_grad()

                # Forward UNet
                mask_pred = self.model(X)                   # mask_pred: [B,1,R,Theta]

                # Compute weak loss
                loss, loss_dict = self.criterion(
                    mask_pred,
                    constraints_batch,
                    X
                )

                # Backpropagation
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * X.size(0)


            mean_train_loss = running_loss / len(self.dataLoader.dataset)
            train_losses.append(mean_train_loss)

            print(f"✅ Epoch {epoch+1} | Train Loss: {mean_train_loss:.4f}")

        # Sauvegarde des poids
        create_folder(f"{self.results_path}/_Weights")
        torch.save(self.model.state_dict(), f"{self.results_path}/_Weights/model.pt")

        # Sauvegarde de la courbe
        plt.plot(train_losses, label='Train')
        plt.legend()
        plt.title("Learning curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(f"{self.results_path}/learning_curve.png")
        plt.close()

    def loadWeights(self, modelPath=None, filename="model.pt"):
        """
        Charge les poids du modèle à partir d'un fichier donné.
        
        Args:
            modelPath (str or None): chemin du dossier contenant le sous-dossier '_Weights'.
                                    Si None, on utilise self.results_path.
            filename (str): nom du fichier de poids (par défaut 'model.pt')
        """
        # --- Détermination du chemin complet ---
        base_path = modelPath if modelPath is not None else self.results_path 
        weights_path = os.path.join(base_path, "_Weights", filename)

        # --- Vérification de l'existence du fichier ---
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"❌ Aucun fichier de poids trouvé à: {weights_path}")

        # Charger le state_dict directement
        try:
            state_dict = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"✅ Poids chargés depuis: {weights_path}")
        except Exception as e:
            print(f"⚠️ Erreur lors du chargement des poids: {e}")
    

    def visualize_samples(self, n_samples=20):
        """
        Sauvegarde n_samples figures dans exp/data_visualize.
        Chaque figure contient les 5 images (t-1, t, t+1, Δ1, Δ2)
        avec les overlays CME comme dans les sanity checks.
        """

        # Sélection dataset
        dataset = self.dataSet

        # Dossier output
        out_dir = os.path.join(self.results_path, "data_visualize")
        os.makedirs(out_dir, exist_ok=True)

        n_samples = min(n_samples, len(dataset))

        # Tirage aléatoire
        import random
        rng = random.Random()
        indices = rng.sample(range(len(dataset)), n_samples)

        # TITRES
        channel_titles = ["t-1", "t", "t+1", "Δ1", "Δ2"]

        for fig_id, idx in enumerate(indices):

            sample = dataset[idx]
            img_stack = sample["image"]            # (C,H,W)
            constraints = sample["constraints"]    # liste de CME
            time = sample["time"]                  # timestamp éventuel

            C, H, W = img_stack.shape

            # CME overlay
            has_cme = len(constraints) > 0
            if has_cme:
                theta_min = constraints[0]["theta_min"]
                theta_max = constraints[0]["theta_max"]
                pa = constraints[0]["pa"]
                da = constraints[0]["da"]
            else:
                theta_min = theta_max = None
                pa = da = 0.0

            # Couleur du cadre
            frame_color = "lime" if has_cme else "gray"

            # Créer figure
            fig, axes = plt.subplots(
                1, 5,
                figsize=(20, 4),
                squeeze=False
            )
            axes = axes[0]

            # ----------------------------------------------------
            # AFFICHAGE des 5 IMAGES avec OVERLAY CME
            # ----------------------------------------------------
            for k in range(5):
                ax = axes[k]
                img = img_stack[k].numpy()

                ax.imshow(img, cmap="gray", vmin=0, vmax=1)

                # Overlay CME si elle existe
                if has_cme:
                    # masque transparent
                    overlay = np.zeros((H, W, 4))      # RGBA

                    # range angulaire
                    tmin = int(theta_min) % W
                    tmax = int(theta_max) % W

                    if tmin <= tmax:
                        overlay[:, tmin:tmax, 1] = 1.0   # canal G = 1
                        overlay[:, tmin:tmax, 3] = 0.25  # alpha
                    else:
                        # cas θ_min > θ_max (wrap around)
                        overlay[:, tmin:, 1] = 1.0
                        overlay[:, tmin:, 3] = 0.25
                        overlay[:, :tmax, 1] = 1.0
                        overlay[:, :tmax, 3] = 0.25

                    ax.imshow(overlay)

                ax.set_title(channel_titles[k], fontsize=10, color=frame_color)
                ax.axis("off")

                # Bordure
                for spine in ax.spines.values():
                    spine.set_edgecolor(frame_color)
                    spine.set_linewidth(2)

            # ----------------------------------------------------
            # TITRE GLOBAL
            # ----------------------------------------------------
            fig.suptitle(
                f"Sample {fig_id:03d} | CME={int(has_cme)} | pa={pa:.1f}° | da={da:.1f}° | t={time}",
                fontsize=14,
                fontweight="bold",
                color=frame_color
            )

            fig.tight_layout(rect=[0, 0, 1, 0.92])

            # ----------------------------------------------------
            # SAVE only
            # ----------------------------------------------------
            out_path = os.path.join(out_dir, f"sample_{fig_id:03d}.png")
            fig.savefig(out_path, dpi=120)
            plt.close(fig)


    def test(self, n_samples=16, save_vis=True, n_inspect=3, save_pred_masks=True, n_save_masks=8):
        """
        Test weak-supervised segmentation model on polaire LASCO images.

        NEW:
        - Save 'n_save_masks' predicted masks into:
            results/exps/<exp_name>/predicted_masks/
        for later gradient analysis.
        """

        self.model.eval()
        all_losses = []

        # NEW: mask prediction save directory
        pred_save_dir = os.path.join(self.results_path, "predicted_masks")
        if save_pred_masks:
            os.makedirs(pred_save_dir, exist_ok=True)
            saved_count = 0

        # --- mask stats ---
        mask_stats = {
            "mean": [], "std": [], "min": [], "max": [],
            "active_pct": [], "angular_var": [], "mask_distance": []
        }
        last_mask = None

        # --- visualization folder ---
        vis_dir = os.path.join(self.results_path, "visualizations")
        if save_vis:
            os.makedirs(vis_dir, exist_ok=True)

        # -----------------------------------------------
        # Collect samples (test=entire dataset here)
        # -----------------------------------------------
        samples = []
        for i, batch in enumerate(self.dataLoader):
            for j in range(batch["image"].size(0)):
                samples.append({
                    "image": batch["image"][j:j+1],    # [1,1,R,Θ]
                    "constraints": batch["constraints"][j],
                    "time": batch["time"][j]
                })
                if len(samples) >= n_samples:
                    break
            if len(samples) >= n_samples:
                break

        # -----------------------------------------------
        # Run through collected samples
        # -----------------------------------------------
        with torch.no_grad():
            for i, sample in enumerate(samples):

                X = sample["image"].to(self.device)
                constraints = [sample["constraints"]]  

                # ---- forward ----
                mask_pred = self.model(X)
                loss, loss_dict = self.criterion(mask_pred, constraints, X)
                all_losses.append(loss.item())

                mask_np = mask_pred[0,0].cpu().numpy()

                # ---- angular profile ----
                A = mask_np.mean(axis=0)
                target = self.criterion.build_target(constraints[0], device="cpu").numpy()

                # ----------------------------
                # NEW: SAVE PREDICTED MASK
                # ----------------------------
                if save_pred_masks and saved_count < n_save_masks:
                    torch.save({
                        "mask_pred": mask_pred[0,0].cpu(),      # [R,Θ]
                        "constraints": constraints[0],           
                        "image": sample["image"][0].cpu(),      # [1,R,Θ]
                        "time": sample["time"]
                    }, os.path.join(pred_save_dir, f"pred_{saved_count:03d}.pt"))

                    saved_count += 1

                # ----------------------------
                # MASK STATISTICS
                # ----------------------------
                mask_stats["mean"].append(mask_np.mean())
                mask_stats["std"].append(mask_np.std())
                mask_stats["min"].append(mask_np.min())
                mask_stats["max"].append(mask_np.max())
                mask_stats["active_pct"].append((mask_np > 0.2).mean())
                mask_stats["angular_var"].append(np.var(A))

                if last_mask is not None:
                    mask_stats["mask_distance"].append(np.mean(np.abs(last_mask - mask_np)))
                last_mask = mask_np.copy()

                # ----------------------------
                # VISUALISATIONS
                # ----------------------------
                if i < n_inspect or save_vis:
                    fig, axs = plt.subplots(3, 1, figsize=(10, 10))

                    axs[0].imshow(X[0,0].cpu(), cmap="gray", aspect="auto")
                    axs[0].set_title(f"Image polaire — {sample['time']}")

                    axs[1].imshow(mask_np, cmap="inferno", aspect="auto")
                    axs[1].set_title("Masque prédit")

                    axs[2].plot(A, label="A(theta) prédiction")
                    axs[2].plot(target, label="target(theta)", linestyle="--")
                    axs[2].set_title("Profil angulaire (prédiction vs cible)")
                    axs[2].set_xlabel("Angle (°)")
                    axs[2].legend()

                    plt.tight_layout()
                    if save_vis:
                        fig.savefig(os.path.join(vis_dir, f"inspect_{i:03d}.png"), dpi=150)
                    plt.close(fig)

        # -----------------------------------------------
        # Résultats globaux du test
        # -----------------------------------------------
        mean_test_loss = np.mean(all_losses)

        print("\n🎯 Test terminé")
        print(f"   ➤ Test Loss = {mean_test_loss:.4f}")

        print("\n📊 Mask Statistics:")
        print(f"   Mean(mask)       : {np.mean(mask_stats['mean']):.4f}")
        print(f"   Std(mask)        : {np.mean(mask_stats['std']):.4f}")
        print(f"   Min(mask)        : {np.mean(mask_stats['min']):.4f}")
        print(f"   Max(mask)        : {np.mean(mask_stats['max']):.4f}")
        print(f"   Active % (>0.2)  : {np.mean(mask_stats['active_pct']):.4f}")
        print(f"   Angular variance : {np.mean(mask_stats['angular_var']):.4f}")

        if len(mask_stats["mask_distance"]) > 0:
            print(f"   Mask distance(mean abs diff) : {np.mean(mask_stats['mask_distance']):.6f}")

        print()
        return mean_test_loss



