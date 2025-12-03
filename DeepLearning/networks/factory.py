from networks.models.unet import WeakCMEUNet
from networks.models.segnet import CMESegNet
from networks.losses.cme_losses import (
    CMELoss,
    WeakCMECompositeLoss,
    CMELossAngularProfileMSE,
    CMELossAngularProfileMSE_V2,
    CMELossAngularProfileMSE_V3,
    CMELossAngularProfileMSE_V4,
)

def get_model(name, **kwargs):
    if name == "WeakCMEUNet":
        return WeakCMEUNet(**kwargs)
    elif name == "CMESegNet":
        return CMESegNet(**kwargs)
    else:
        raise ValueError(f"Unknown model name: {name}")

def get_loss(name, **kwargs):
    if name == "CMELoss":
        return CMELoss(**kwargs)
    elif name == "WeakCMECompositeLoss":
        return WeakCMECompositeLoss(**kwargs)
    elif name == "CMELossAngularProfileMSE":
        return CMELossAngularProfileMSE(**kwargs)
    elif name == "CMELossAngularProfileMSE_V2":
        return CMELossAngularProfileMSE_V2(**kwargs)
    elif name == "CMELossAngularProfileMSE_V3":
        return CMELossAngularProfileMSE_V3(**kwargs)
    elif name == "CMELossAngularProfileMSE_V4":
        return CMELossAngularProfileMSE_V4(**kwargs)
    else:
        raise ValueError(f"Unknown loss name: {name}")
