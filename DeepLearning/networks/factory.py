from networks.models.unet import WeakCMEUNet
from networks.models.segnet import CMESegNet
from networks.losses.cme_losses import (
    CMELossAngularProfileMSE,
    CMELossAngularProfileWasserstein,
    CMELossAngularProfileMSE_V4
)

def get_model(name, **kwargs):
    if name == "WeakCMEUNet":
        return WeakCMEUNet(**kwargs)
    elif name == "CMESegNet":
        return CMESegNet(**kwargs)
    else:
        raise ValueError(f"Unknown model name: {name}")

def get_loss(name, **kwargs):
    if name == "CMELossAngularProfileMSE":
        return CMELossAngularProfileMSE(**kwargs)
    elif name == "CMELossAngularProfileWasserstein":
        return CMELossAngularProfileWasserstein(**kwargs)
    elif name == "CMELossAngularProfileMSE_V4":
        return CMELossAngularProfileMSE_V4(**kwargs)
    else:
        raise ValueError(f"Unknown loss name: {name}")
