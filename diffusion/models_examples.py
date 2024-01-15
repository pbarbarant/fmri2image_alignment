# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from utils import plot_images
from modules import UNet_conditional
from ddpm import Diffusion


metadata = pd.read_csv(
    "/data/parietal/store3/work/pbarbara/fmri2image_alignment/data/NSD/dataset_aligned/dataset.csv"
)

device = "cuda"
model = UNet_conditional(mri_dim=2).to(device)
model = nn.DataParallel(model)
ckpt = torch.load(
    "/data/parietal/store3/work/pbarbara/fmri2image_alignment/models/DDPM_conditional/ckpt.pt"
)
model.load_state_dict(ckpt)
diffusion = Diffusion(img_size=64, device=device)
n = 16
image_paths = metadata["image_path"].values.tolist()
fmri_paths = metadata["fmri_path"].values.tolist()

fmris = np.vstack(
    [np.array(np.load(fmri_paths[i])) for i in range(n)],
)
fmris = torch.Tensor(fmris).to(device)
x = diffusion.sample(model, n, fmris)
plot_images(x)
