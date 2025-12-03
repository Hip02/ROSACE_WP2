import argparse
import os
import sys
import yaml
import numpy as np
import torch

# Local imports
from utils.utils import LascoC2ImagesDataset, MyDataLoader
from utils.sanity_checkers.sanity_check_polar import PolarTransformChecker
from utils.sanity_checkers.sanity_check_head import CMEHeadChecker
from utils.sanity_checkers.sanity_check_gradient_loss import sanity_check_gradient_analysis
from utils.tools.cme_video_maker import CMEVideoMaker
from utils.tools.cme_video_pred_maker import CMEVideoPredMaker
from networks.model import Network

# ============================================================
#               ON / OFF PARAMETERS
# ============================================================
DO_SANITY_CHECKS = True
TRAIN_MODEL = False
TEST_MODEL = False
MAKE_VIDEO = False
# ============================================================

# Dataset paths
dataset_path = "../Dataset"
images_path = os.path.join(dataset_path, "data_lasco_c2_png")
labels_path = os.path.join(dataset_path, "cmes_lz_20220101_20221231.csv")


# ============================================================
#               PARSE CONFIG FILE
# ============================================================
parser = argparse.ArgumentParser(description="Run experiment")
parser.add_argument("--config", type=str, default="exp_list/example_config.yaml")
args = parser.parse_args()

with open(args.config, "r") as f:
    config = yaml.safe_load(f)

exp_name = config["experiment"]["name"]
param = config["parameters"]

print(f"\n=== Running experiment: {exp_name} ===")
print(f"Parameters:\n{param}\n")


# ============================================================
#               DATA LOADER
# ============================================================
dataLoader = MyDataLoader(
    root_dir=images_path,
    labels_file=labels_path,
    param=param
)

device = "mps"


# ============================================================
#               MODEL INITIALIZATION
# ============================================================
net = Network(
    dataLoader=dataLoader,
    exp_name=exp_name,
    param=param,
    device=device
)

weights_path = os.path.join("results/exps", exp_name, "_Weights", "model.pt")


# ============================================================
#               SANITY CHECKS
# ============================================================
if DO_SANITY_CHECKS:
    print("\n=== SANITY CHECKS ===")

    print("\n→ Gradient Loss Check")
    sanity_check_gradient_analysis(net, exp_name=exp_name, param=param)

    print("\nSanity checks completed. Exiting.")
    sys.exit(0)



# ============================================================
#               TRAINING OR LOADING
# ============================================================
if TRAIN_MODEL:
    print("\n=== TRAINING MODEL ===")
    net.train()
else:
    if os.path.exists(weights_path):
        print("\n=== LOADING EXISTING WEIGHTS ===")
        net.loadWeights()
    else:
        print("\nNo weights found. Enable TRAIN_MODEL to train.")
        sys.exit(0)


# ============================================================
#               TESTING
# ============================================================
if TEST_MODEL:
    print("\n=== TESTING MODEL ===")
    try:
        net.test()
    except Exception as e:
        print(f"⚠️ Test failed: {e}")


# ============================================================
#               VIDEO PREDICTION
# ============================================================
if MAKE_VIDEO:
    print("\n=== MAKING PREDICTION VIDEO ===")

    try:
        os.makedirs("results/videos", exist_ok=True)

        videoMaker = CMEVideoPredMaker(
            dataLoader=dataLoader,                # reuse same dataloader
            DatasetClass=LascoC2ImagesDataset,    # dataset for frame-by-frame
            network=net.model,
            device=device
        )

        videoMaker.make_video(
            year=2022,
            month=3,
            out_path=f"results/videos/cme_pred_{exp_name}.mp4",
            show_background=False                # option from your previous request
        )

        print(f"Video saved in results/videos/cme_pred_{exp_name}.mp4")

    except Exception as e:
        print(f"Could not make video: {e}")

print("\n=== DONE ===\n")
