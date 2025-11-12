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
    "polar_transform": True
}

dataLoader = MyDataLoader(root_dir=images_path, labels_file=labels_path, param=param)

net = Network(dataLoader, exp_name="firstRun", param=param, epochs=3)

#net.visualize(mode="train")
#net.train()
net.loadWeights()
net.test()