import argparse
import os
import sys
import yaml
import numpy as np
from utils.utils import LascoC2ImagesDataset, MyDataLoader
from utils.sanity_checkers.sanity_check_polar import PolarTransformChecker
from utils.sanity_checkers.sanity_check_head import CMEHeadChecker
from networks.model import Network

dataset_path = "/Users/hippolytehilgers/Desktop/ROSACE/Project_ROSACE/WP2/Dataset/"
images_path = os.path.join(dataset_path, "data_lasco_c2_png")
labels_path = os.path.join(dataset_path, "cmes_lz_20220101_20221231.csv")

# Manual change of parameters
param = {
    "polar_transform": True
}

#sanityChecker = PolarTransformChecker(MyDataLoader, LascoC2ImagesDataset, images_path, labels_path)
#sanityChecker.check()


# TO RUN AFTER THE GYM
checker = CMEHeadChecker(H=363, W=360)

checker.test(pa=45, aw=30)        # Un cas
checker.test_basic()              # Tests généraux
#checker.test_extremes()           # PA extrêmes
#checker.test_widths()             # Largeurs différentes


#dataLoader = MyDataLoader(root_dir=images_path, labels_file=labels_path, param=param)

#net = Network(dataLoader, exp_name="firstRun", param=param, epochs=3)

#net.visualize(mode="train")
#net.train()
#net.loadWeights()
#net.test()