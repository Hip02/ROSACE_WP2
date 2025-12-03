import torch
import torch.nn as nn
import torch.nn.functional as F

def build_target(n_bins, constraints, device, gaussian=False, sigma=10.0):
    """
    Build target angular profile T(theta) from CME constraints.

    Args:
        constraints : list of dicts containing theta_min, theta_max
        gaussian    : if True → gaussian-smoothed target
        sigma       : std of Gaussian (in degrees)
    """
    theta = torch.arange(n_bins, device=device).float()
    T = torch.zeros(n_bins, device=device)

    if not constraints:
        return T

    # -------------------------------
    # BINARY TARGET (base version)
    # -------------------------------
    if gaussian is False:
        for c in constraints:
            tmin = int(c["theta_min"]) % n_bins
            tmax = int(c["theta_max"]) % n_bins

            if tmin <= tmax:
                T[tmin:tmax+1] = 1.0
            else:
                T[tmin:] = 1.0
                T[:tmax+1] = 1.0

        return T

    # ----------------------------------------------
    # GAUSSIAN TARGET (smooth circular distribution)
    # ----------------------------------------------
    for c in constraints:
        tmin = float(c["theta_min"]) % n_bins
        tmax = float(c["theta_max"]) % n_bins

        # Compute circular center
        if tmin <= tmax:
            center = 0.5 * (tmin + tmax)
        else:
            center = (tmin + (tmax + n_bins)) / 2.0
            if center >= n_bins:
                center -= n_bins

        # Circular distance
        dist = torch.minimum(
            (theta - center).abs(),
            n_bins - (theta - center).abs()
        )

        # Gaussian bump around the CME center
        T += torch.exp(-0.5 * (dist / sigma) ** 2)

    # Clamping to avoid >1 when multiple CME overlap
    T = torch.clamp(T, 0.0, 1.0)
    return T

class CMELossAngularProfileMSE(nn.Module):
    """
    Very simple angular loss:
        Loss = MSE( A_pred(theta) , T(theta) )

    where:
      A_pred(theta) = radial mean of mask prediction
      T(theta)      = binary target profile derived from constraints
    """

    def __init__(self, gaussian_target, sigma_target, n_bins=360):
        super().__init__()
        self.n_bins = n_bins
        self.gaussian_target = gaussian_target
        self.sigma_target = sigma_target
        self.lambda_ang = 1.0
        self.crit = nn.MSELoss()

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
                T = build_target(self.n_bins, constraints_batch[b], device, gaussian=self.gaussian_target, sigma=self.sigma_target)
            else:
                T = torch.zeros(Theta, device=device)

            # Simple MSE between profiles
            L_ang = self.crit(A, T)

            total_loss += self.lambda_ang * L_ang
            logs["ang"] += L_ang.item()

        total_loss /= B
        logs = {k: v/B for k,v in logs.items()}

        return total_loss, logs


class CMELossAngularProfileWasserstein(nn.Module):
    """
    Computes the 1D Wasserstein-1 (Earth Mover's Distance)
    between the angular profile A(theta) = mean(mask_pred[:,theta])
    and a constraint-derived target profile T(theta).
    """

    def __init__(self, gaussian_target, sigma_target, n_bins=360):
        super().__init__()
        self.n_bins = n_bins
        self.gaussian_target = gaussian_target
        self.sigma_target = sigma_target
        self.lambda_ang = 1.0

    # ------------------------------------------------------------------
    # Compute EMD / Wasserstein-1 distance between A and T (+ circularity prise en compte)
    # ------------------------------------------------------------------
    def wasserstein_circular(self, A, T):
        """
        Compute circular Wasserstein-1 distance (Earth Mover’s Distance on a ring).

        Args:
            A, T: tensors [Theta], non-negative, not necessarily normalized.
        """
        # normalize to a distribution
        A = A / (torch.sum(A) + 1e-6)
        T = T / (torch.sum(T) + 1e-6)

        # linear CDFs
        cdf_A = torch.cumsum(A, dim=0)
        cdf_T = torch.cumsum(T, dim=0)

        diff = cdf_A - cdf_T

        # circular optimal shift k = median(diff)
        k = torch.median(diff)

        # circular Wasserstein
        W = torch.mean(torch.abs(diff - k))

        return W

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, mask_pred, constraints_batch, images=None):
        """
        mask_pred : [B,1,R,Theta]
        """
        B,_,R,Theta = mask_pred.shape
        device = mask_pred.device

        total_loss = 0.
        logs = {"w1": 0.}

        for b in range(B):
            mask = mask_pred[b,0]            # [R,Theta]
            A = mask.mean(dim=0)             # angular profile

            # create profile target
            if len(constraints_batch[b]) > 0:
                T = build_target(self.n_bins, constraints_batch[b], device, gaussian=self.gaussian_target, sigma=self.sigma_target)
            else:
                T = torch.zeros(Theta, device=device)

            # Wasserstein-1 distance
            L_w1 = self.wasserstein_circular(A, T)

            total_loss += self.lambda_ang * L_w1
            logs["w1"] += L_w1.item()

        total_loss /= B
        logs = {k: v/B for k, v in logs.items()}

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
