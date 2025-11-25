import argparse
import os
import sys
import yaml
import numpy as np
from utils.utils import LascoC2ImagesDataset, MyDataLoader
from utils.sanity_checkers.sanity_check_polar import PolarTransformChecker
from utils.sanity_checkers.sanity_check_head import CMEHeadChecker
from utils.sanity_checkers.sanity_check_gradient_loss import sanity_check_gradient_analysis
from utils.tools.cme_video_maker import CMEVideoMaker
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

net = Network(dataLoader, exp_name="sanity_gradient_MSE_V4", param=param, device="mps")

sanity_check_gradient_analysis(net, exp_name="sanity_gradient_MSE_V4")