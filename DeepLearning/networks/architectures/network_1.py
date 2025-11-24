import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# -----------------------------------------------------------
# Réseau complet : segmentation + extraction de paramètres CME
# -----------------------------------------------------------


class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.net(x)


class WeakCMEUNet(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()

        # Encoder
        self.enc1 = UNetBlock(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = UNetBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = UNetBlock(64, 128)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = UNetBlock(128, 256)

        # Decoder
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = UNetBlock(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = UNetBlock(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = UNetBlock(64, 32)

        # Final segmentation map
        self.out_conv = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        # Encoding
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2))
        x4 = self.enc4(self.pool3(x3))

        # Decoding
        y3 = self.up3(x4)
        y3 = self.dec3(torch.cat([y3, x3], dim=1))

        y2 = self.up2(y3)
        y2 = self.dec2(torch.cat([y2, x2], dim=1))

        y1 = self.up1(y2)
        y1 = self.dec1(torch.cat([y1, x1], dim=1))

        mask = torch.sigmoid(self.out_conv(y1))  # segmentation

        return mask



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



class CMELoss(nn.Module):
    """
    Loss pour [has_cme, PA_deg, AW_deg] avec pondération λ.
    """

    def __init__(self, lambda_has=1.0, lambda_pa=1.0, lambda_aw=1.0):
        super().__init__()
        self.lambda_has = lambda_has
        self.lambda_pa  = lambda_pa
        self.lambda_aw  = lambda_aw

    def circular_mse_deg(self, pred_deg, target_deg):
        pred_rad = pred_deg * torch.pi / 180.0
        targ_rad = target_deg * torch.pi / 180.0
        diff = torch.atan2(
            torch.sin(pred_rad - targ_rad),
            torch.cos(pred_rad - targ_rad)
        )
        return (diff ** 2).mean()

    def forward(self, pred, target):
        """
        pred, target : (B, 3)
        dans l'ordre [has, pa_deg, da_deg]
        """

        pred_has = pred[:, 0]
        pred_pa  = pred[:, 1]
        pred_aw  = pred[:, 2]

        targ_has = target[:, 0]
        targ_pa  = target[:, 1]
        targ_aw  = target[:, 2]

        # --- has_cme ---
        L_has = F.binary_cross_entropy(
            pred_has.clamp(1e-6, 1 - 1e-6),
            targ_has
        )

        # --- PA circulaire ---
        L_pa = self.circular_mse_deg(pred_pa, targ_pa)

        # --- AW normalisé ---
        L_aw = F.mse_loss(pred_aw / 360.0, targ_aw / 360.0)

        # --- Pondération ---
        L = (
            self.lambda_has * L_has +
            self.lambda_pa  * L_pa +
            self.lambda_aw  * L_aw
        )

        return L, {
            "has": L_has.item(),
            "pa":  L_pa.item(),
            "aw":  L_aw.item()
        }
    

class WeakCMECompositeLoss(nn.Module):
    """
    Weak-supervised segmentation loss for LASCO CME masks in polar coordinates.
    Includes: angular, fg, bg, contrast, radial, photo, motion, static, no-CME suppression.

    channel 0 → I(t-1)
    channel 1 → I(t)
    channel 2 → I(t+1)
    channel 3 → Δ(t-1→t)
    channel 4 → Δ(t→t+1)
    """

    def __init__(self, n_bins=360, param=None):
        super().__init__()

        p = param if param is not None else {}

        # Par défaut : tout à 0 sauf lambda_ang = 1
        self.lambda_ang      = p.get("lambda_ang", 1.0)
        self.lambda_seg      = p.get("lambda_seg", 0.0)
        self.lambda_fg       = p.get("lambda_fg", 0.0)
        self.lambda_bg       = p.get("lambda_bg", 0.0)
        self.lambda_contrast = p.get("lambda_contrast", 0.0)
        self.lambda_radial   = p.get("lambda_radial", 0.0)
        self.lambda_motion   = p.get("lambda_motion", 0.0)
        self.lambda_static   = p.get("lambda_static", 0.0)
        self.lambda_no_cme   = p.get("lambda_no_cme", 0.0)

        self.use_neighbor_diff = p.get("use_neighbor_diff", False)

        self.bce = nn.BCELoss()

        self.n_bins = n_bins

    # -------------------------------------------------------------
    ### NEW
    def dice_loss(self, pred, target, eps=1e-6):
        pred = torch.sigmoid(pred)
        num = 2 * (pred * target).sum()
        den = pred.sum() + target.sum()
        return 1 - (num + eps) / (den + eps)

    # -------------------------------------------------------------
    def build_target(self, constraints, device):
        target = torch.zeros(self.n_bins, device=device)
        for c in constraints:
            tmin = int(c["theta_min"]) % self.n_bins
            tmax = int(c["theta_max"]) % self.n_bins
            if tmin <= tmax:
                target[tmin:tmax+1] = 1.0
            else:
                target[tmin:] = 1.0
                target[:tmax+1] = 1.0
        return target

    # -------------------------------------------------------------
    def forward(self, mask_pred, constraints_batch, images):
        """
        mask_pred : [B,1,R,Theta]
        images    : [B,5,R,Theta]
        """
        B = mask_pred.size(0)
        device = mask_pred.device

        L_total = 0.0

        logs = {
            "ang": 0.0,
            "fg": 0.0,
            "bg": 0.0,
            "contrast": 0.0,
            "radial": 0.0,
            "motion": 0.0,
            "static": 0.0,
            "no_cme": 0.0,
            "seg": 0.0
        }

        for b in range(B):

            mask = mask_pred[b, 0]     # [R,Theta]

            # ----- channels ------
            # Two supported input formats:
            # - When using neighbor diffs: images is a stack of 5 tensors
            #   [I(t-1), I(t), I(t+1), dI1, dI2]
            # - When not using neighbors: images is a single tensor [I(t)]
            C = images.size(1)

            if self.use_neighbor_diff:
                if C < 5:
                    raise ValueError(f"use_neighbor_diff=True requires images with 5 channels, got {C}")
                # use provided neighbor diffs
                I_t  = images[b, 1]           # [R,Theta]
                dI_1 = torch.abs(images[b, 3])
                dI_2 = torch.abs(images[b, 4])
                dI = dI_1 + dI_2
                dI_norm = dI / (dI.max() + 1e-6)
            else:
                if C != 1:
                    raise ValueError(f"use_neighbor_diff=False requires images with a single channel, got {C}")
                # single-frame input -> no motion info
                I_t = images[b, 0]            # [R,Theta]
                dI_norm = torch.zeros_like(I_t, device=device, dtype=I_t.dtype)

            if len(constraints_batch[b]) > 0:
                target = self.build_target(constraints_batch[b], device)
            else:
                target = torch.zeros(self.n_bins, device=device)

            # ------------------------------------------
            # 0) NO CME SUPPRESSION
            # ------------------------------------------
            if self.lambda_no_cme > 0.0:
                if len(constraints_batch[b]) == 0:
                    L_no_cme = mask.mean()

                    A = mask.mean(0)
                    L_ang_noCME = (A ** 2).mean()
                else:
                    L_no_cme = torch.tensor(0.0, device=device)
                    L_ang_noCME = torch.tensor(0.0, device=device)
            else:
                L_no_cme = torch.tensor(0.0, device=device)
                L_ang_noCME = torch.tensor(0.0, device=device)

            # ------------------------------------------
            # 1) Angular supervision (only if CME)
            # ------------------------------------------
            if self.lambda_ang > 0.0 and len(constraints_batch[b]) == 0:
                A = mask.mean(dim=0)
                L_ang = F.mse_loss(A, target)

                fg_mask = target == 1
                bg_mask = target == 0
            else:
                L_ang = torch.tensor(0.0, device=device)
                fg_mask = torch.zeros(self.n_bins, dtype=torch.bool, device=device)
                bg_mask = torch.zeros(self.n_bins, dtype=torch.bool, device=device)

            # ------------------------------------------
            # 2) Foreground loss
            # ------------------------------------------
            if self.lambda_fg > 0.0:
                if fg_mask.any():
                    L_fg = (1 - mask[:, fg_mask]).mean()
                else:
                    L_fg = torch.tensor(0.0, device=device)
            else:
                L_fg = torch.tensor(0.0, device=device)

            # ------------------------------------------
            # 3) Background loss
            # ------------------------------------------
            if self.lambda_bg > 0.0:
                if bg_mask.any():
                    L_bg = mask[:, bg_mask].mean()
                else:
                    L_bg = torch.tensor(0.0, device=device)
            else:
                L_bg = torch.tensor(0.0, device=device)

            # ------------------------------------------
            # 4) Contrastive
            # ------------------------------------------
            if self.lambda_contrast > 0.0:
                if fg_mask.any() and bg_mask.any():
                    fg_val = mask[:, fg_mask].mean()
                    bg_val = mask[:, bg_mask].mean()
                    L_contrast = torch.relu(bg_val - fg_val + self.contrast_margin)
                else:
                    L_contrast = torch.tensor(0.0, device=device)
            else:
                L_contrast = torch.tensor(0.0, device=device)

            # ------------------------------------------
            # 5) Radial smoothness
            # ------------------------------------------
            if self.lambda_radial > 0.0:
                L_radial = torch.mean((mask[1:, :] - mask[:-1, :])**2)
            else:
                L_radial = torch.tensor(0.0, device=device)

            # ------------------------------------------
            # 6) Motion-based photometric loss
            #     mask ~ dI (pas ~ I(t))
            # ------------------------------------------
            if self.lambda_motion > 0.0:
                if fg_mask.any():
                    L_motion = F.mse_loss(mask[:, fg_mask], dI_norm[:, fg_mask])
                else:
                    L_motion = torch.tensor(0.0, device=device)
            else:
                L_motion = torch.tensor(0.0, device=device)

            # ------------------------------------------
            # 7) Static suppression
            #     punitions des activations dans zones immobiles
            # ------------------------------------------
            if self.lambda_static > 0.0:
                L_static = (mask * (1 - dI_norm)).mean()
            else:
                L_static = torch.tensor(0.0, device=device)

            # ------------------------------------------
            # 8) NEW — Segmentation loss
            # ------------------------------------------
            if self.lambda_seg > 0.0:
                # (a) construire le masque synthétique [R,Theta]
                target_mask = target.unsqueeze(0).repeat(mask.shape[0], 1)

                # (b) Dice + BCE
                L_seg = (
                    self.bce(mask, target_mask) +
                    self.dice_loss(mask, target_mask)
                )
            else:
                L_seg = torch.tensor(0.0, device=device)

            # ------------------------------------------
            # TOTAL
            # ------------------------------------------

            L = (
                self.lambda_ang      * L_ang +
                self.lambda_fg       * L_fg +
                self.lambda_bg       * L_bg +
                self.lambda_contrast * L_contrast +
                self.lambda_radial   * L_radial +

                # NEW :
                self.lambda_motion   * L_motion +
                self.lambda_static   * L_static +
                self.lambda_no_cme   * (L_no_cme + L_ang_noCME) +
                self.lambda_seg      * L_seg
                )

            L_total += L

            logs["ang"]      += L_ang.item()
            logs["fg"]       += L_fg.item()
            logs["bg"]       += L_bg.item()
            logs["contrast"] += L_contrast.item()
            logs["radial"]   += L_radial.item()
            logs["motion"]   += L_motion.item()
            logs["static"]   += L_static.item()
            logs["no_cme"]   += (L_no_cme.item() + L_ang_noCME.item())
            logs["seg"]      += L_seg.item()

        L_total /= B
        for k in logs:
            logs[k] /= B

        return L_total, logs


class CMELoss2(nn.Module):
    """
    Circular angular loss using trigonometric embedding.

    For each angular bin θ:
      v_pred(θ)   = A_pred(θ) * [cosθ, sinθ]
      v_target(θ) = T(θ)       * [cosθ, sinθ]

    Loss = SmoothL1( v_pred , v_target )

    This naturally respects circular geometry: no discontinuity at 0°/360°.
    """

    def __init__(self, n_bins=360, lambda_ang=1.0):
        super().__init__()
        self.n_bins = n_bins
        self.lambda_ang = lambda_ang
        self.crit = nn.SmoothL1Loss()

        # Precompute angles
        theta_deg = torch.arange(n_bins)
        theta_rad = theta_deg * torch.pi / 180
        self.cos = torch.cos(theta_rad).unsqueeze(0)   # [1,Theta]
        self.sin = torch.sin(theta_rad).unsqueeze(0)   # [1,Theta]

    def build_target(self, constraints, device):
        t = torch.zeros(self.n_bins, device=device)
        for c in constraints:
            tmin = int(c["theta_min"]) % self.n_bins
            tmax = int(c["theta_max"]) % self.n_bins
            if tmin <= tmax:
                t[tmin:tmax+1] = 1.0
            else:
                t[tmin:] = 1.0
                t[:tmax+1] = 1.0
        return t

    def forward(self, mask_pred, constraints_batch, images=None):
        B,_,R,Theta = mask_pred.shape
        device = mask_pred.device

        cos = self.cos.to(device)  # [1,Theta]
        sin = self.sin.to(device)

        total_loss = 0.0
        logs = {"ang": 0.0}

        for b in range(B):
            mask = mask_pred[b,0]          # [R,Theta]
            A = mask.mean(dim=0)           # angular profile [Theta]

            if len(constraints_batch[b]) > 0:
                T = self.build_target(constraints_batch[b], device)
            else:
                T = torch.zeros(Theta, device=device)

            # Project onto circle
            v_pred   = torch.stack([A * cos, A * sin], dim=0)     # [2,Theta]
            v_target = torch.stack([T * cos, T * sin], dim=0)     # [2,Theta]

            L_ang = self.crit(v_pred, v_target)

            total_loss += self.lambda_ang * L_ang
            logs["ang"] += L_ang.item()

        total_loss /= B
        logs = {k: v/B for k,v in logs.items()}

        return total_loss, logs
