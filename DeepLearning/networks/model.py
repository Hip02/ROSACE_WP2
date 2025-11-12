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
from networks.architectures.network_1 import CMESegNet

# ========================
# 1. UTILITAIRES
# ========================
def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)

def create_folder(path: str):
    if not os.path.exists(path):
        os.makedirs(path)

# ========================
# 2. CLASSE ENTRAÎNEMENT/TEST
# ========================
class Network:
    def __init__(self, dataLoader, param=None, exp_name="Unnamed", device="cpu", lr=1e-3, batch_size=16, epochs=5):
        set_seed(42)
        self.dataLoader = dataLoader
        self.device = device
        self.epochs = epochs
        self.results_path = "exp_list/" + exp_name
        self.batchSize = batch_size
        self.lr = lr
        
        create_folder(self.results_path)

        # Data Loaders
        self.dataSetTrain = LascoC2ImagesDataset(self.dataLoader, mode='train', param=param)
        self.dataSetVal = LascoC2ImagesDataset(self.dataLoader, mode='val', param=param)
        self.dataSetTest = LascoC2ImagesDataset(self.dataLoader, mode='test', param=param)

        self.trainDataLoader = DataLoader(self.dataSetTrain, batch_size=self.batchSize, shuffle=True, num_workers=0)
        self.valDataLoader = DataLoader(self.dataSetVal, batch_size=self.batchSize, shuffle=False, num_workers=0)
        self.testDataLoader = DataLoader(self.dataSetTest, batch_size=self.batchSize, shuffle=False, num_workers=0)

        for X, y in self.trainDataLoader:
            print("Train data loader X shape:", X.shape)
            print("Train data loader Y shape::", y.shape)
            break

        # Model + Optimizer + Loss
        self.model = CMESegNet(in_channels=1).to(device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def train(self):
        train_losses, val_losses = [], []

        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0

            for X, y in tqdm(self.trainDataLoader, desc=f"Epoch {epoch+1}/{self.epochs}"):
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                mask_pred, outputs = self.model(X)
                loss = self.criterion(outputs, y)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * X.size(0)

            mean_train_loss = running_loss / len(self.trainDataLoader.dataset)
            train_losses.append(mean_train_loss)

            val_loss = self.validate()
            val_losses.append(val_loss)

            print(f"✅ Epoch {epoch+1} | Train Loss: {mean_train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Sauvegarde des poids
        create_folder(f"{self.results_path}/_Weights")
        torch.save(self.model.state_dict(), f"{self.results_path}/_Weights/model.pt")

        # Sauvegarde des courbes
        plt.plot(train_losses, label='Train')
        plt.plot(val_losses, label='Val')
        plt.legend()
        plt.title("Learning curves")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(f"{self.results_path}/learning_curves.png")
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
    

    def visualize(self, n_samples=16, mode="train", save=False):
        """
        Affiche une grille 4x4 d'images du dataset choisi (train/val/test)
        avec les 3 infos du label : has_cme, pa_deg, da_deg, et la date de l'image.
        """

        assert mode in ["train", "val", "test"], "mode doit être 'train', 'val' ou 'test'"

        # Sélection du dataset
        if mode == "train":
            dataset = self.dataSetTrain
        elif mode == "val":
            dataset = self.dataSetVal
        else:
            dataset = self.dataSetTest

        n_samples = min(n_samples, len(dataset))
        rng = random.Random()
        indices = rng.sample(range(len(dataset)), n_samples)

        n_cols = 4
        n_rows = (n_samples + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 12))
        axes = axes.flatten()

        for ax, idx in zip(axes, indices):
            # On suppose que le dataset renvoie (image, label)
            image, label = dataset[idx]
            has_cme, pa_deg, da_deg = label.tolist()

            # Extraire la date à partir du nom de fichier
            img_path = dataset.image_paths[dataset.data_indices[idx]]
            filename = os.path.basename(img_path)

            try:
                parts = filename.split("_")
                date_str = f"{parts[1]}-{parts[2]}-{parts[3]} {parts[4]}:{parts[5]}:{parts[6].split('.')[0]}"
                date = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
                date_label = date.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                date_label = "unknown date"

            # Image
            img_np = image.squeeze().numpy()
            ax.imshow(img_np, cmap='gray', vmin=0, vmax=1)
            ax.axis("off")

            # Couleur selon présence de CME
            color = "lime" if has_cme > 0.5 else "gray"

            # Titre formaté proprement
            ax.set_title(
                f"has_cme: {int(has_cme)}\n"
                f"pa: {pa_deg:.1f}° | da: {da_deg:.1f}°\n"
                f"{date_label}",
                fontsize=8.5,
                color=color,
                fontweight="bold",
                linespacing=1.2
            )

            # Bordure colorée
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(2)

        # Cacher les cases vides si < 16
        for ax in axes[n_samples:]:
            ax.axis("off")

        plt.suptitle(f"LASCO C2 samples ({mode} set)", fontsize=14, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if save:
            out_dir = os.path.join(self.results_path, "visualizations")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"sample_grid_{mode}.png")
            plt.savefig(out_path, dpi=150)
            print(f"💾 Visualization saved to: {out_path}")

        plt.show()

    def validate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for X, y in self.valDataLoader:
                X, y = X.to(self.device), y.to(self.device)
                mask_pred, outputs = self.model(X)
                loss = self.criterion(outputs, y)
                total_loss += loss.item() * X.size(0)
        return total_loss / len(self.valDataLoader.dataset)

    def test(self, n_samples=16, save_vis=True, n_inspect=3):
        """
        Évalue le modèle sur le test set :
        (1) Calcule les MSE des 3 paramètres (has_cme, pa_deg, da_deg)
        (2) Visualise pour plusieurs exemples (image + masque)
        (3) Inspecte n_inspect prédictions complètes (masques + valeurs)
        (4) Trace des nuages de points GT vs PR pour évaluer la distribution des prédictions
        """

        self.model.eval()
        total_loss = 0
        all_preds, all_labels = [], []

        # --- Évaluation quantitative ---
        with torch.no_grad():
            for X, y in tqdm(self.testDataLoader, desc="Testing"):
                X, y = X.to(self.device), y.to(self.device)
                mask_pred, outputs = self.model(X)
                loss = self.criterion(outputs, y)
                total_loss += loss.item() * X.size(0)
                all_preds.append(outputs.cpu())
                all_labels.append(y.cpu())

        preds = torch.cat(all_preds, dim=0)
        labels = torch.cat(all_labels, dim=0)

        mse_has = torch.mean((preds[:, 0] - labels[:, 0]) ** 2).item()
        mse_pa  = torch.mean((preds[:, 1] - labels[:, 1]) ** 2).item()
        mse_da  = torch.mean((preds[:, 2] - labels[:, 2]) ** 2).item()
        total_mse = (mse_has + mse_pa + mse_da) / 3

        print(f"\n📏 Test results:")
        print(f" - MSE has_cme : {mse_has:.4f}")
        print(f" - MSE pa_deg  : {mse_pa:.4f}")
        print(f" - MSE da_deg  : {mse_da:.4f}")
        print(f" - Mean MSE     : {total_mse:.4f}\n")

        # --- Visualisation qualitative : images et masques ---
        n_samples = min(n_samples, len(self.dataSetTest))
        indices = random.sample(range(len(self.dataSetTest)), n_samples)

        n_cols_pairs = 4
        n_rows = math.ceil(n_samples / n_cols_pairs)
        n_cols = n_cols_pairs * 2

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.6 * n_cols_pairs, 3.1 * n_rows))
        axes = axes.flatten()
        im_ref = None

        for i, idx in enumerate(indices):
            X, y = self.dataSetTest[idx]
            X = X.unsqueeze(0).to(self.device)

            with torch.no_grad():
                mask_pred, outputs = self.model(X)

            image = X.squeeze().cpu().numpy()
            mask_pred = mask_pred.squeeze().cpu().numpy()
            has_cme_true, pa_true, da_true = y.tolist()
            has_cme_pred, pa_pred, da_pred = outputs.squeeze().cpu().tolist()

            # Image originale
            ax_img = axes[2 * i]
            ax_img.imshow(image, cmap='gray', vmin=0, vmax=1)
            ax_img.set_title(
                f"GT → has={has_cme_true:.1f}\npa={pa_true:.0f}°, da={da_true:.0f}°",
                fontsize=8.5, color="white", backgroundcolor="black", pad=2
            )
            ax_img.axis("off")
            for spine in ax_img.spines.values():
                spine.set_edgecolor("white")
                spine.set_linewidth(1.2)

            # Masque prédit
            ax_mask = axes[2 * i + 1]
            im_ref = ax_mask.imshow(mask_pred, cmap="magma_r")
            ax_mask.set_title(
                f"PR → has={has_cme_pred:.1f}\npa={pa_pred:.0f}°, da={da_pred:.0f}°",
                fontsize=8.5, color="white", backgroundcolor="black", pad=2
            )
            ax_mask.axis("off")
            for spine in ax_mask.spines.values():
                spine.set_edgecolor("white")
                spine.set_linewidth(1.2)

        for ax in axes[2 * n_samples:]:
            ax.axis("off")

        if im_ref is not None:
            cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
            fig.colorbar(im_ref, cax=cbar_ax, label="Predicted CME Mask Intensity")

        plt.suptitle("CME Mask Predictions — Test Set", fontsize=14, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 0.9, 0.96])

        if save_vis:
            out_dir = os.path.join(self.results_path, "visualizations")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, "test_predictions_grid.png")
            plt.savefig(out_path, dpi=150)
            print(f"💾 Visualization saved to: {out_path}")

        plt.show()

        # --- Scatter plots : corrélation GT ↔ PR ---
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        titles = ["has_cme", "pa_deg", "da_deg"]
        for i, (ax, title) in enumerate(zip(axs, titles)):
            ax.scatter(labels[:, i], preds[:, i], alpha=0.6, s=15)
            ax.plot([labels[:, i].min(), labels[:, i].max()],
                    [labels[:, i].min(), labels[:, i].max()],
                    'r--', linewidth=1)
            ax.set_title(f"{title} — GT vs Pred", fontsize=10)
            ax.set_xlabel("GT")
            ax.set_ylabel("Predicted")
            ax.grid(alpha=0.3)

        plt.suptitle("Scatter Plots of Predictions", fontsize=13, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

        # --- Inspection de plusieurs prédictions complètes ---
        print("\n🔍 Detailed inspection of several predictions:")
        inspect_indices = random.sample(range(len(self.dataSetTest)), n_inspect)
        np.set_printoptions(precision=4, suppress=True, linewidth=150)

        for idx_inspect in inspect_indices:
            X, y = self.dataSetTest[idx_inspect]
            X = X.unsqueeze(0).to(self.device)
            with torch.no_grad():
                mask_pred, outputs = self.model(X)

            mask_np = mask_pred.squeeze().cpu().numpy()
            has_cme_pred, pa_pred, da_pred = outputs.squeeze().cpu().tolist()
            has_cme_true, pa_true, da_true = y.tolist()

            print(f"\n--- Sample {idx_inspect} ---")
            print(f"GT → has={has_cme_true:.3f}, pa={pa_true:.2f}°, da={da_true:.2f}°")
            print(f"PR → has={has_cme_pred:.3f}, pa={pa_pred:.2f}°, da={da_pred:.2f}°")
            print(f"Mask shape: {mask_np.shape}")
            print(f"Mask min: {mask_np.min():.5f} | max: {mask_np.max():.5f} | mean: {mask_np.mean():.5f}")
            print(mask_np)

        return {
            "mse_has": mse_has,
            "mse_pa": mse_pa,
            "mse_da": mse_da,
            "mean_mse": total_mse
        }
