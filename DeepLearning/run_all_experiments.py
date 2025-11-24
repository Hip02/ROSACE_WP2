import os
from networks.model import Network
from utils.utils import MyDataLoader
from utils.utils import LascoC2ImagesDataset

dataset_path = "/Users/hippolytehilgers/Desktop/ROSACE/Project_ROSACE/WP2/Dataset/"
images_path = os.path.join(dataset_path, "data_lasco_c2_png")
labels_path = os.path.join(dataset_path, "cmes_lz_20220101_20221231.csv")

# Liste des expériences à tester
"""
experiment_list = [
    #("A1_ang_only",      {"lambda_ang":1}),
    ("A1_ang_only_low_lr_no_neighbours", {"lambda_ang":1, "lr": 1e-4, "use_neighbor_diff": False}),
    ("A3_seg_only",      {"lambda_seg":1}),
    ("B4_fg_bg_contrast",{"lambda_fg":1, "lambda_bg":1, "lambda_contrast":1}),
    ("C3_motion_static", {"lambda_motion":1, "lambda_static":1}),
    ("D5_full",          {"lambda_ang":1, "lambda_width":1, "lambda_seg":1,
                          "lambda_fg":1,"lambda_bg":1,"lambda_contrast":1,
                          "lambda_radial":0.1,"lambda_motion":1,"lambda_static":0.5})
]
"""

experiment_list = [
    ("CMELoss2_test_1_frame", {"lr": 1e-3, "use_neighbor_diff": False}),
    ("CMELoss2_test_5_frames", {"lr": 1e-3, "use_neighbor_diff": True})
    ]

# Constantes globales
base_param = {
    "polar_transform": True,
    "subsampling_factor": 10
}

def run_all(device="mps"):
    for exp_name, lambdas in experiment_list:
        print(f"\n==============================")
        print(f"🚀 Lancement de : {exp_name}")
        print(f"==============================\n")

        # Merge base params + lambdas
        param = dict(base_param)
        param.update(lambdas)

        dataLoader = MyDataLoader(
            root_dir=images_path,
            labels_file=labels_path,
            param=param
        )

        net = Network(
            dataLoader,
            exp_name=exp_name,
            param=param,
            epochs=3,
            device=device
        )

        net.train()
        net.loadWeights()
        net.test()

if __name__ == "__main__":
    run_all(device="mps")
