import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# -----------------------------------------------------------
# Réseau complet : segmentation + extraction de paramètres CME
# -----------------------------------------------------------

class CMESegNet(nn.Module):
    """
    Réseau CNN de segmentation + tête régressive pour extraire :
    - has_cme
    - principal_angle
    - angular_width
    """

    def __init__(self, in_channels=1):
        super().__init__()

        # --- 1️⃣ Encoder ---
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 128x128
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 64x64
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
        )

        # --- 2️⃣ Decoder (remonte vers la taille originale) ---
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2),  # 128x128
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 2, stride=2),  # 256x256
            nn.ReLU(),
            nn.Conv2d(16, 1, 1),                      # carte de segmentation binaire
            nn.Sigmoid()
        )

        # --- 3️⃣ Head régressive ---
        self.regression_head = CMEParamExtractor()

    def forward(self, x):
        """
        Args:
            x: tensor (B, 1, H, W)
        Returns:
            mask: (B, 1, H, W)
            outputs: (B, 3) -> [has_cme, pa_deg, da_deg]
        """
        features = self.encoder(x)
        mask = self.decoder(features)
        outputs = self.regression_head(mask)
        return mask, outputs


# -----------------------------------------------------------
# Head différentiable : extrait (has_cme, pa_deg, da_deg)
# -----------------------------------------------------------

class CMEParamExtractor(nn.Module):
    """
    Head différentiable pour extraire (has_cme, principal_angle, angular_width)
    à partir d’un masque de segmentation 2D (B, 1, H, W).
    """

    def __init__(self):
        super().__init__()

    def forward(self, mask):
        """
        Args:
            mask: tensor (B, 1, H, W), valeurs entre [0, 1]
        Returns:
            out: (B, 3) = [has_cme, pa_deg, da_deg]
        """
        B, _, H, W = mask.shape

        # --- has_cme ---
        has_cme = mask.view(B, -1).mean(dim=1, keepdim=True)  # moyenne globale

        # --- principal_angle ---
        # Profil angulaire (moyenne sur la dimension radiale)
        density = mask.mean(dim=2).squeeze(1)  # (B, W)
        angles = torch.linspace(0, 360, W, device=mask.device).unsqueeze(0)  # (1, W)
        weights = density / (density.sum(dim=1, keepdim=True) + 1e-8)
        pa = (weights * angles).sum(dim=1, keepdim=True)  # barycentre pondéré

        # --- angular_width ---
        # Version différentiable du seuil
        thresh = 0.3
        alpha = 20.0
        soft_active = torch.sigmoid(alpha * (density - thresh))
        da = soft_active.sum(dim=1, keepdim=True) * (360.0 / W)

        out = torch.cat([has_cme, pa, da], dim=1)
        return out