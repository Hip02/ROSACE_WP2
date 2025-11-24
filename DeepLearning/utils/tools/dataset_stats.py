import os
import torch
from torch.utils.data import Dataset, DataLoader
from tabulate import tabulate
from utils import LascoC2ImagesDataset, MyDataLoader

# ---------------------------------------------------------
# 2) CONFIGURATION
# ---------------------------------------------------------

dataset_path = "/Users/hippolytehilgers/Desktop/ROSACE/Project_ROSACE/WP2/Dataset/"
images_path = os.path.join(dataset_path, "data_lasco_c2_png")
labels_path = os.path.join(dataset_path, "cmes_lz_20220101_20221231.csv")

param = None   # si tu utilises un paramètre supplémentaire


# ---------------------------------------------------------
# 3) Instancier le DataLoader et le Dataset (100% train)
# ---------------------------------------------------------

dataLoader = MyDataLoader(
    root_dir=images_path,
    labels_file=labels_path,
    param=param
)

# On met 100% du dataset dans "train"
dataset = LascoC2ImagesDataset(
    dataloader=dataLoader,
    mode="train",
    param=param,
    split_ratio=(1.0, 0.0, 0.0)
)

# Pas de shuffle, on veut juste lire tout
loader = DataLoader(dataset, batch_size=1, shuffle=False)


# ---------------------------------------------------------
# 4) Analyse du dataset
# ---------------------------------------------------------

total = 0
with_cme = 0
without_cme = 0

pa_values = []
aw_values = []

for X, y in loader:
    total += 1

    has_cme = y[0, 0].item()

    if has_cme >= 0.5:
        with_cme += 1
        pa_values.append(y[0, 1].item())
        aw_values.append(y[0, 2].item())
    else:
        without_cme += 1

# ---------------------------------------------------------
# 5) Tableau récapitulatif
# ---------------------------------------------------------

stats_table = [
    ["Nombre total d'images", total],
    ["Images avec CME", with_cme, f"{100*with_cme/total:.2f}%"],
    ["Images sans CME", without_cme, f"{100*without_cme/total:.2f}%"]
]

print("\n📊 STATISTIQUES DU DATASET\n")
print(tabulate(stats_table,
               headers=["Catégorie", "Nombre", "Pourcentage"],
               tablefmt="fancy_grid"))

# ---------------------------------------------------------
# 6) Optionnel : résumé PA / AW
# ---------------------------------------------------------

if len(pa_values) > 0:
    pa_tensor = torch.tensor(pa_values)
    aw_tensor = torch.tensor(aw_values)

    print("\n📐 PA (principal angle) — statistiques :")
    print(f" min: {pa_tensor.min():.2f}°")
    print(f" max: {pa_tensor.max():.2f}°")
    print(f" mean: {pa_tensor.mean():.2f}°")
    print(f" std: {pa_tensor.std():.2f}°")

    print("\n📏 AW (angular width) — statistiques :")
    print(f" min: {aw_tensor.min():.2f}°")
    print(f" max: {aw_tensor.max():.2f}°")
    print(f" mean: {aw_tensor.mean():.2f}°")
    print(f" std: {aw_tensor.std():.2f}°")
