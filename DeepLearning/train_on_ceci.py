import argparse
import os
import sys
import yaml
import numpy as np
from utils.utils import LascoC2ImagesDataset, MyDataLoader
from networks.model import Network

dataset_path = "/Users/hippolytehilgers/Desktop/ROSACE/Project_ROSACE/WP2/Dataset/"
images_path = os.path.join(dataset_path, "data_lasco_c2_png")
labels_path = os.path.join(dataset_path, "cmes_lz_20220101_20221231.csv")

# Manual change of parameters
param = {
    "polar_transform": True,
    "subsampling_factor": 10,
    "use_neighbor_diff": True
}

dataLoader = MyDataLoader(root_dir=images_path, labels_file=labels_path, param=param)

net = Network(dataLoader, exp_name="Loss_V4_test", param=param, device="mps")

net.loadWeights()   

videoMaker = CMEVideoPredMaker(
    dataLoader=dataLoader,
    DatasetClass=LascoC2ImagesDataset,
    network=net.model,
    device="mps"
)

videoMaker.make_video(year=2022, month=3, out_path="results/videos/cme_pred_2022_01.mp4", show_background=False)
