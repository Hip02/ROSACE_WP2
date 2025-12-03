import os
from networks.model import Network
from utils.utils import MyDataLoader
from utils.utils import LascoC2ImagesDataset

dataset_path = "./Dataset"
images_path = os.path.join(dataset_path, "data_lasco_c2_png")
labels_path = os.path.join(dataset_path, "cmes_lz_20220101_20221231.csv")

# Liste des expériences à tester
experiment_list = [
    #("CMELoss2_test_1_frame", {"lr": 1e-3, "use_neighbor_diff": False}),
    #("CMELoss2_test_5_frames", {"lr": 1e-3, "use_neighbor_diff": True})
    #("CropBottomTrue", {"crop_polar_bottom": True}),
    #("CropBottomFalse", {"crop_polar_bottom": False}),
    ("Loss_V4_test", {"use_neighbor_diff": True}),
    ("Loss_V4_no_neighbor_diff", {"use_neighbor_diff": False})
    ]

# Constantes globales
base_param = {
    "polar_transform": True,
    "subsampling_factor": 5,
    "epochs": 3
}

import torch
def run_all(device="mps"):
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cuda" if torch.cuda.is_available() else "cpu"

    for exp_name, options in experiment_list:
        print(f"\n==============================")
        print(f"🚀 Lancement de : {exp_name}")
        print(f"==============================\n")

        # Merge base params + lambdas
        param = dict(base_param)
        param.update(options)

        dataLoader = MyDataLoader(
            root_dir=images_path,
            labels_file=labels_path,
            param=param
        )

        net = Network(
            dataLoader,
            exp_name=exp_name,
            param=param,
            device=device
        )

        net.train()
        net.loadWeights()
        net.test()

if __name__ == "__main__":
    run_all(device="mps")
