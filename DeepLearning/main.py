import argparse
import os
import sys
import yaml
import numpy as np
import torch
from utils.utils import LascoC2ImagesDataset, MyDataLoader
from utils.sanity_checkers.sanity_check_polar import PolarTransformChecker
from utils.sanity_checkers.sanity_check_head import CMEHeadChecker
from utils.sanity_checkers.sanity_check_gradient_loss import sanity_check_gradient_analysis
from utils.tools.cme_video_maker import CMEVideoMaker
from utils.tools.cme_video_pred_maker import CMEVideoPredMaker
from networks.model import Network

dataset_path = "./Dataset"
images_path = os.path.join(dataset_path, "data_lasco_c2_png")
labels_path = os.path.join(dataset_path, "cmes_lz_20220101_20221231.csv")

# Parse arguments
parser = argparse.ArgumentParser(description='Run experiment')
parser.add_argument('--config', type=str, default="exp_list/example_config.yaml", help='Path to config file')
args = parser.parse_args()

# Load config
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

exp_name = config["experiment"]["name"]
param = config["parameters"]

print(f"Running experiment: {exp_name}")
print(f"Parameters: {param}")

dataLoader = MyDataLoader(root_dir=images_path, labels_file=labels_path, param=param)

device = "cuda" if torch.cuda.is_available() else "cpu"
net = Network(dataLoader, exp_name=exp_name, param=param, device=device)

# Check if weights exist before loading
if os.path.exists(os.path.join("results/exps", exp_name, "_Weights", "model.pt")):
    net.loadWeights()
else:
    print("No weights found, starting training.")
    net.train()

try:
    videoMaker = CMEVideoPredMaker(
        dataLoader=dataLoader,
        DatasetClass=LascoC2ImagesDataset,
        network=net.model,
        device=device
    )

    # Create output directory
    os.makedirs("results/videos", exist_ok=True)
    videoMaker.make_video(year=2022, month=3, out_path=f"results/videos/cme_pred_{exp_name}.mp4", show_background=False)
except Exception as e:
    print(f"Could not make video: {e}")
