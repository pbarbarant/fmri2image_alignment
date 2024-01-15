# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from ddpm import Diffusion
from modules import UNet_conditional
from PIL import Image
from tqdm import tqdm


def get_fmri(fmri_path):
    fmri = np.load(fmri_path)
    return fmri


def get_image(image_path):
    image = Image.open(image_path)
    return image


def get_paths(subject):
    metadata = pd.read_csv(
        "/data/parietal/store3/work/pbarbara/fmri2image_alignment/data/NSD/dataset_unaligned/dataset.csv"
    )
    metadata = metadata[metadata["shared1000"] == True]
    metadata = metadata[metadata["subject"] == subject]
    fmri_paths = metadata["fmri_path"].values.tolist()
    image_paths = metadata["image_path"].values.tolist()
    return fmri_paths, image_paths


def load_model(ckpt_path, device):
    model = UNet_conditional(mri_dim=2).to(device)
    model = nn.DataParallel(model)
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt)
    return model


def retrieve_all_subjects(N=10):
    subjects = list(range(1, 9))
    fmri_all_subjects = []
    for subject in subjects:
        fmri_paths, image_paths = get_paths(subject)
        fmris = np.vstack([get_fmri(fmri_paths[i]) for i in range(N)])
        fmri_all_subjects.append(fmris)

    fmri_all_subjects = np.array(fmri_all_subjects)
    fmri_all_subjects = torch.Tensor(fmri_all_subjects).to(device)
    return fmri_all_subjects, image_paths


if __name__ == "__main__":
    device = "cuda"
    ckpt_path = "/data/parietal/store3/work/pbarbara/fmri2image_alignment/models/unaligned_embeddings/ckpt2.pt"
    model = load_model(ckpt_path, device)
    diffusion = Diffusion(img_size=64, device=device)
    N = 100
    fmri_all_subjects, image_paths = retrieve_all_subjects(N=N)
    print("Successfully retrieved all subjects test data")

    for i in tqdm(range(N)):
        x = diffusion.sample(model, 8, fmri_all_subjects[:, i, ...])
        # Plot the obtained images and the original image
        plt.figure(figsize=(20, 20))
        plt.subplot(1, 9, 1)
        plt.imshow(get_image(image_paths[i]))
        plt.axis("off")
        plt.title("GT")
        for j in range(8):
            plt.subplot(1, 9, j + 2)
            # Add the name of the subject
            plt.title(f"Subject {j + 1}")
            # Show the RGB image
            plt.imshow(x[j].permute(1, 2, 0).cpu().numpy())
            plt.axis("off")

        # Add the word "Unaligned data" as legend on the left vertically
        plt.subplot(1, 9, 1)
        plt.text(
            -0.1,
            0.5,
            "Unaligned data",
            fontsize=15,
            horizontalalignment="center",
            verticalalignment="center",
            rotation=90,
            transform=plt.gca().transAxes,
        )
        # Stich the images together
        plt.subplots_adjust(wspace=0, hspace=0)
        # Save the image in the results folder
        out_path = Path(
            "/data/parietal/store3/work/pbarbara/fmri2image_alignment/figures/unaligned_visual_eval"
        )
        plt.savefig(
            out_path / f"{i}.png",
            bbox_inches="tight",
        )
