import torch
import torch.nn as nn

# -----------------------------------------------------------
# Head différentiable : extrait (has_cme, pa_deg, da_deg)
# -----------------------------------------------------------

class CMEParamExtractor(nn.Module):
    """
    Head différentiable pour extraire (has_cme, principal_angle, angular_width)
    à partir d’un masque polaire (B, 1, H, W).
    """

    def __init__(self):
        super().__init__()

        # paramètres du seuil flou
        self.thresh = 0.3
        self.alpha = 20.0

    def forward(self, mask):
        """
        Args:
            mask: tensor (B, 1, H, W) in [0,1]
        Returns:
            out: (B, 3) = [has_cme, pa_deg, da_deg]
        """
        B, _, H, W = mask.shape

        # -------------------------------------------------------
        # 1) Profil angulaire (moyenne sur les rayons)
        # -------------------------------------------------------
        density = mask.mean(dim=2).squeeze(1)   # (B, W)

        # Valeur minimale du "soft threshold" quand density = 0
        # soft_min = sigmoid(alpha * (0 - thresh))
        soft_min = torch.sigmoid(
            torch.tensor(self.alpha * (0.0 - self.thresh),
                         device=mask.device, dtype=mask.dtype)
        )

        # Activation douce : sigmoid(alpha*(density - thresh))
        soft = torch.sigmoid(self.alpha * (density - self.thresh))

        # On enlève la valeur de base soft_min → soft=0 quand masque vide
        soft = torch.clamp(soft - soft_min, min=0.0)

        # -------------------------------------------------------
        # 2) has_cme — OR floue basée sur la somme des activations
        #    has_cme = 1 - exp(-sum(soft))
        # -------------------------------------------------------
        act = soft.sum(dim=1, keepdim=True)           # (B,1)
        has_cme = 1.0 - torch.exp(-act)               # dans [0,1]
        has_cme = has_cme.clamp(0, 1)

        # -------------------------------------------------------
        # 3) PA = barycentre circulaire
        #    atan2( Σ w·sinθ , Σ w·cosθ )
        # -------------------------------------------------------
        angles = torch.linspace(0, 2 * torch.pi, W, device=mask.device).unsqueeze(0)
        w = density / (density.sum(dim=1, keepdim=True) + 1e-8)

        sin_term = (w * torch.sin(angles)).sum(dim=1, keepdim=True)
        cos_term = (w * torch.cos(angles)).sum(dim=1, keepdim=True)

        pa_rad = torch.atan2(sin_term, cos_term) % (2 * torch.pi)
        pa_deg = pa_rad * (180.0 / torch.pi)

        # -------------------------------------------------------
        # 4) AW = proportion d'activation, normalisée pour éviter
        #    la sous-estimation due au soft-threshold
        #
        #    da_deg = 360 * sum(soft) / sum_max
        # -------------------------------------------------------
        # maximum théorique si density=1 partout
        tmp = torch.tensor(self.alpha * (1.0 - self.thresh),
                           device=mask.device, dtype=mask.dtype)
        max_soft = torch.sigmoid(tmp) - soft_min     # valeur soft pour density=1
        max_possible = max_soft * W                  # somme max possible

        da_deg = (soft.sum(dim=1, keepdim=True) / (max_possible + 1e-8)) * 360.0
        da_deg = da_deg.clamp(0.0, 360.0)

        # -------------------------------------------------------
        # Résultat final
        # -------------------------------------------------------
        out = torch.cat([has_cme, pa_deg, da_deg], dim=1)
        return out

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
