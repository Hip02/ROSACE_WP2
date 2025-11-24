import argparse
import os
import sys
import yaml
import numpy as np
from utils.utils import LascoC2ImagesDataset, MyDataLoader
from utils.sanity_checkers.sanity_check_polar import PolarTransformChecker
from utils.sanity_checkers.sanity_check_head import CMEHeadChecker
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

net = Network(dataLoader, exp_name="firstRun", param=param, epochs=3, device="mps")

net.visualize_samples(n_samples=30, mode="train")
#net.train()
#net.loadWeights()
#net.test()




"""
maker = CMEVideoMaker(
        DataLoaderClass=MyDataLoader,
        DatasetClass=LascoC2ImagesDataset,
        images_path=images_path,
        labels_path=labels_path
    )

maker.make_video(year=2022, month=2, out_path="cme_2022_02.mp4", fps=10)
maker.make_video(year=2022, month=3, out_path="cme_2022_03.mp4", fps=10)
"""


# ===============================================================
# 1) Sanity Checker Polar Transform
# ===============================================================
#sanityChecker = PolarTransformChecker(MyDataLoader, LascoC2ImagesDataset, images_path, labels_path)
#sanityChecker.check()

# ===============================================================
# 2) Sanity Checker CME Head
# ===============================================================

#checker = CMEHeadChecker(H=363, W=360)

#checker.test(pa=45, aw=30)        # Un cas de CME simple
#checker.test(pa=0.0, aw=0.0)      # Un cas sans CME
#checker.test_noisy()              # Test avec bruit
#checker.test(pa=270, aw=60, noise_sigma=0.3)         # Très bruité
#checker.test(pa=0.0, aw=0.0, noise_sigma=0.3)       # Sans CME très bruité. => devrait rester négatif (ne fonctionne pas encore)
#checker.test_basic()              # Tests généraux
#checker.test_extremes()           # PA extrêmes
#checker.test_widths()             # Largeurs différentes
