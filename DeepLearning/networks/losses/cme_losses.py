import torch
import torch.nn as nn
import torch.nn.functional as F

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


class CMELossAngularProfileMSE(nn.Module):
    """
    Very simple angular loss:
        Loss = MSE( A_pred(theta) , T(theta) )

    where:
      A_pred(theta) = radial mean of mask prediction
      T(theta)      = binary target profile derived from constraints
    """

    def __init__(self, n_bins=360, lambda_ang=1.0):
        super().__init__()
        self.n_bins = n_bins
        self.lambda_ang = lambda_ang
        self.crit = nn.MSELoss()

    def build_target(self, constraints, device):
        """
        Build a binary angular profile T(θ) from constraints.
        """
        T = torch.zeros(self.n_bins, device=device)

        for c in constraints:
            tmin = int(c["theta_min"]) % self.n_bins
            tmax = int(c["theta_max"]) % self.n_bins

            if tmin <= tmax:
                T[tmin:tmax+1] = 1.0
            else:  # wrap around 360° -> 0°
                T[tmin:] = 1.0
                T[:tmax+1] = 1.0

        return T

    def forward(self, mask_pred, constraints_batch, images=None):
        """
        mask_pred : [B,1,R,Theta]
        constraints_batch : list of lists of constraints
        """
        B,_,R,Theta = mask_pred.shape
        device = mask_pred.device

        total_loss = 0.0
        logs = {"ang": 0.0}

        for b in range(B):
            mask = mask_pred[b,0]        # [R,Theta]
            A = mask.mean(dim=0)         # Angular profile [Theta]

            # Build target profile
            if len(constraints_batch[b]) > 0:
                T = self.build_target(constraints_batch[b], device)
            else:
                T = torch.zeros(Theta, device=device)

            # Simple MSE between profiles
            L_ang = self.crit(A, T)

            total_loss += self.lambda_ang * L_ang
            logs["ang"] += L_ang.item()

        total_loss /= B
        logs = {k: v/B for k,v in logs.items()}

        return total_loss, logs
    

class CMELossAngularProfileMSE_V2(nn.Module):
    """
    Angular MSE with:
      - Gaussian soft target
      - Weighted MSE (more error when far from CME)
    """

    def __init__(self, n_bins=360, lambda_ang=1.0, sigma=10.0, alpha_weight=2.0):
        super().__init__()
        self.n_bins = n_bins
        self.lambda_ang = lambda_ang
        self.sigma = sigma
        self.alpha_weight = alpha_weight  # amplifie les erreurs loin de la cible

    # ============ UTILITAIRES ============

    def circular_distance(self, theta, center):
        """Distance angulaire circulaire."""
        return torch.minimum(
            (theta - center).abs(),
            360 - (theta - center).abs()
        )

    def gaussian_1D(self, theta, center, sigma):
        """Gaussian circulaire."""
        dist = self.circular_distance(theta, center)
        return torch.exp(-0.5 * (dist / sigma) ** 2)

    # ============ CIBLE GAUSSIENNE ============

    def build_target(self, constraints, device):
        """
        Création d'un profil T(θ) fait de gaussiennes.
        """
        theta = torch.arange(self.n_bins, device=device).float()
        T = torch.zeros(self.n_bins, device=device)

        for c in constraints:
            tmin = float(c["theta_min"])
            tmax = float(c["theta_max"])

            # centre approximatif
            if tmin <= tmax:
                center = 0.5 * (tmin + tmax)
            else:
                # wrap: e.g. 350 → 10
                center = (tmin + (tmax + 360)) / 2.0
                if center >= 360:
                    center -= 360

            # Gaussian lissée autour du centre
            T += self.gaussian_1D(theta, center, self.sigma)

        # Normalisation optionnelle (évite des pics si 2 CME)
        T = torch.clamp(T, 0.0, 1.0)
        return T

    # ============ FORWARD ============

    def forward(self, mask_pred, constraints_batch, images=None):
        """
        mask_pred: [B,1,R,Theta]
        """
        B, _, R, Theta = mask_pred.shape
        device = mask_pred.device

        total_loss = 0.0
        logs = {"ang": 0.0}

        # angles
        theta = torch.arange(Theta, device=device).float()

        for b in range(B):
            mask = mask_pred[b, 0]             # [R,Theta]
            A = mask.mean(dim=0)               # Profil angulaire

            # ---- Build target ----
            if len(constraints_batch[b]) > 0:
                T = self.build_target(constraints_batch[b], device)
            else:
                T = torch.zeros(Theta, device=device)

            # ---- Weighted MSE ----
            # Poids angulaires = plus l'angle est loin du centre CME, plus w est grand
            if len(constraints_batch[b]) > 0:
                # compute a combined distance weight
                w = torch.zeros_like(theta)
                for c in constraints_batch[b]:
                    center = float(c["theta_min"] + (c["theta_max"] - c["theta_min"]) % 360) / 2.0
                    dist = self.circular_distance(theta, center)
                    w = torch.maximum(w, dist)  # max dist to any CME center

                # poids = 1 + alpha * (dist normalisée)
                w = 1 + self.alpha_weight * (w / 180.0)
            else:
                w = torch.ones_like(theta)

            # Weighted MSE
            L_ang = ((A - T) ** 2 * w).mean()

            total_loss += self.lambda_ang * L_ang
            logs["ang"] += L_ang.item()

        total_loss /= B
        logs = {k: v / B for k, v in logs.items()}
        return total_loss, logs
    

class CMELossAngularProfileMSE_V3(nn.Module):
    """
    Angular profile loss with:
      - Gaussian soft targets
      - Circular distance weighting
      - Smoothness regularization (2nd derivative)
      - Light convolution smoothing of A(theta)
    """

    def __init__(
        self,
        n_bins=360,
        lambda_ang=1.0,
        sigma_target=15.0,
        alpha_weight=2.0,
        lambda_smooth=0.1,
    ):
        super().__init__()
        self.n_bins = n_bins
        self.lambda_ang = lambda_ang
        self.sigma_target = sigma_target    # sigma of Gaussian targets
        self.alpha_weight = alpha_weight    # weight strength for distance
        self.lambda_smooth = lambda_smooth  # strength of A'' regularization

        # convolution kernel for smoothing (light)
        k = torch.tensor([0.25, 0.5, 0.25], dtype=torch.float32)
        self.register_buffer("kernel", k[None, None, :])  # shape [1,1,3]

    # -------------------------------------------------------
    # Utility: circular distance
    # -------------------------------------------------------
    def circ_dist(self, theta, center):
        """returns circular angular distance |θ - μ| mod 360."""
        diff = (theta - center).abs()
        return torch.minimum(diff, 360 - diff)

    # -------------------------------------------------------
    # Gaussian target construction
    # -------------------------------------------------------
    def build_target(self, constraints, device):
        theta = torch.arange(self.n_bins, device=device).float()
        T = torch.zeros_like(theta)

        for c in constraints:
            tmin = float(c["theta_min"])
            tmax = float(c["theta_max"])

            # compute correct circular center
            if tmax >= tmin:
                center = 0.5 * (tmin + tmax)
            else:
                # wrap case (ex: 350 -> 10)
                center = (tmin + (tmax + 360)) / 2.0
                if center >= 360:
                    center -= 360

            # Gaussian around CME center
            dist = self.circ_dist(theta, center)
            T += torch.exp(-0.5 * (dist / self.sigma_target) ** 2)

        # clamp in case of double CME
        return torch.clamp(T, 0.0, 1.0)

    # -------------------------------------------------------
    # Smoothness regularization: || A''(theta) ||^2
    # -------------------------------------------------------
    def smoothness_penalty(self, A):
        # circular second derivative
        A_roll_f = torch.roll(A, -1)
        A_roll_b = torch.roll(A, 1)
        A_second = A_roll_f - 2 * A + A_roll_b
        return (A_second ** 2).mean()

    # -------------------------------------------------------
    # Forward
    # -------------------------------------------------------
    def forward(self, mask_pred, constraints_batch, images=None):
        """
        mask_pred : [B,1,R,Theta]
        """
        B,_,R,Theta = mask_pred.shape
        device = mask_pred.device
        theta = torch.arange(Theta, device=device).float()

        total_loss = 0.0
        logs = {"ang": 0.0, "smooth": 0.0}

        for b in range(B):
            # Angular profile
            mask = mask_pred[b,0]       # [R,Theta]
            A = mask.mean(dim=0)        # [Theta]

            # Light smoothing of A(theta)
            A_smooth = F.conv1d(
                A[None,None,:], 
                self.kernel, 
                padding=1
            )[0,0]

            # Build target T(theta)
            if len(constraints_batch[b]) > 0:
                T = self.build_target(constraints_batch[b], device)
            else:
                T = torch.zeros(Theta, device=device)

            # -------------------------------------------------------
            # Distance-based weight: small near CME, large far away
            # -------------------------------------------------------
            if len(constraints_batch[b]) > 0:
                dmin = None
                for c in constraints_batch[b]:
                    # compute correct center
                    tmin = float(c["theta_min"])
                    tmax = float(c["theta_max"])
                    if tmax >= tmin:
                        center = 0.5 * (tmin + tmax)
                    else:
                        center = (tmin + (tmax + 360)) / 2.0
                        if center >= 360: center -= 360

                    dist = self.circ_dist(theta, center)
                    dmin = dist if dmin is None else torch.minimum(dmin, dist)

                # weights: w = 1 + alpha * (dist normalized)
                w = 1 + self.alpha_weight * (dmin / 180.0)
            else:
                w = torch.ones_like(theta)

            # -------------------------------------------------------
            # Weighted MSE
            # -------------------------------------------------------
            mse = w * (A_smooth - T) ** 2
            L_ang = mse.mean()

            # -------------------------------------------------------
            # Smoothness regularisation
            # -------------------------------------------------------
            L_smooth = self.smoothness_penalty(A_smooth)

            total_loss += self.lambda_ang * L_ang + self.lambda_smooth * L_smooth
            logs["ang"] += L_ang.item()
            logs["smooth"] += L_smooth.item()

        total_loss /= B
        logs = {k: v/B for k,v in logs.items()}
        return total_loss, logs
    

class CMELossAngularProfileMSE_V4(nn.Module):
    """
    Angular profile loss with:
      - Gaussian soft targets (circular)
      - Angular smoothing of A(theta)
      - CME-distance weighting (far from CME -> stronger correction)
      - Error-distance weighting (large |A-T| -> stronger correction)
      - Angular smoothness regularization (second derivative)
    """

    def __init__(
        self,
        n_bins=360,
        lambda_ang=1.0,
        sigma_target=12.0,     # largeur des gaussiennes CME
        alpha_cme=2.0,         # poids distance à la CME
        alpha_error=3.0,       # poids distance erreur |A-T|
        lambda_smooth=0.05,    # régularisation smoothness
    ):
        super().__init__()
        self.n_bins = n_bins
        self.lambda_ang = lambda_ang
        self.sigma_target = sigma_target
        self.alpha_cme = alpha_cme
        self.alpha_error = alpha_error
        self.lambda_smooth = lambda_smooth

        # petit kernel de lissage angulaire
        kernel = torch.tensor([0.25, 0.5, 0.25], dtype=torch.float32)
        self.register_buffer("kernel", kernel[None,None,:])  # shape [1,1,3]

    # ------------------------------------------------------------------
    # Circular distance between angles
    # ------------------------------------------------------------------
    def circ_dist(self, theta, center):
        diff = (theta - center).abs()
        return torch.minimum(diff, 360 - diff)

    # ------------------------------------------------------------------
    # Build soft Gaussian target for 1 sample
    # ------------------------------------------------------------------
    def build_target(self, constraints, device):
        theta = torch.arange(self.n_bins, device=device).float()
        T = torch.zeros_like(theta)

        for c in constraints:
            tmin = float(c["theta_min"])
            tmax = float(c["theta_max"])

            # center of CME interval (circular)
            if tmax >= tmin:
                center = (tmin + tmax) / 2
            else:
                # wrap example: 350 → 10
                center = (tmin + (tmax + 360)) / 2
                if center >= 360:
                    center -= 360

            dist = self.circ_dist(theta, center)
            T += torch.exp(-0.5 * (dist / self.sigma_target)**2)

        return torch.clamp(T, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Angular smoothness regularization (second derivative)
    # ------------------------------------------------------------------
    def smoothness_penalty(self, A):
        A_f = torch.roll(A, -1)
        A_b = torch.roll(A, 1)
        A_second = A_f - 2 * A + A_b
        return (A_second**2).mean()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, mask_pred, constraints_batch, images=None):
        """
        mask_pred : [B,1,R,Theta]
        """
        B,_,R,Theta = mask_pred.shape
        device = mask_pred.device
        theta = torch.arange(Theta, device=device).float()

        total_loss = 0.
        logs = {"ang": 0., "smooth": 0.}

        for b in range(B):
            mask = mask_pred[b,0]       # [R,Theta]
            A = mask.mean(dim=0)        # profil angulaire brut

            # petit lissage pour briser le sawtooth
            A_smooth = F.conv1d(
                A[None,None,:],
                self.kernel,
                padding=1
            )[0,0]

            # ---- Build soft target ----
            if len(constraints_batch[b]) > 0:
                T = self.build_target(constraints_batch[b], device)
            else:
                T = torch.zeros(Theta, device=device)

            # ----------------------------------------------------------
            # CME-distance weighting: small near CME, large far
            # ----------------------------------------------------------
            if len(constraints_batch[b]) > 0:
                dmin = None
                for c in constraints_batch[b]:
                    tmin = float(c["theta_min"])
                    tmax = float(c["theta_max"])

                    # center
                    if tmax >= tmin:
                        center = (tmin + tmax) / 2
                    else:
                        center = (tmin + (tmax + 360)) / 2
                        if center >= 360:
                            center -= 360

                    dist = self.circ_dist(theta, center)
                    dmin = dist if dmin is None else torch.minimum(dmin, dist)

                W_cme = 1 + self.alpha_cme * (dmin / 180.0)
            else:
                W_cme = torch.ones_like(theta)

            # ----------------------------------------------------------
            # Error-distance weighting: strong correction where A!=T
            # ----------------------------------------------------------
            D_err = torch.abs(A_smooth - T)
            W_err = 1 + self.alpha_error * D_err

            # Total weight
            W = W_cme * W_err

            # Weighted MSE
            L_ang = (W * (A_smooth - T)**2).mean()

            # Smoothness penalty
            L_s = self.smoothness_penalty(A_smooth)

            total_loss += self.lambda_ang * L_ang + self.lambda_smooth * L_s
            logs["ang"] += L_ang.item()
            logs["smooth"] += L_s.item()

        total_loss /= B
        logs = {k: v/B for k,v in logs.items()}
        return total_loss, logs
