# %%
import os

import numpy as np
import pandas as pd
import torch
import torchvision
from matplotlib import pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class FmriDataset(Dataset):
    def __init__(self, csv_file, split="train", transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.filtered_data = self.df[
            (self.df["split"] == split) & (self.df["shared1000"] == True)
        ]

    def __len__(self):
        return len(self.filtered_data)

    def __getitem__(self, index):
        # Load paths and labels from the filtered dataframe
        image_path = self.filtered_data.iloc[index]["image_path"]
        if image_path.startswith("/storage"):
            image_path = image_path.replace("/storage", "/data/parietal")
        fmri_path = self.filtered_data.iloc[index]["fmri_path"]
        if fmri_path.startswith("/storage"):
            fmri_path = fmri_path.replace("/storage", "/data/parietal")

        image = Image.open(image_path).convert("RGB")
        fmri_data = torch.from_numpy(np.load(fmri_path))

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        # Return a dictionary containing the samples and labels
        return {"image": image, "fmri": fmri_data}


def get_data(args):
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((64, 64)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_dataset = FmriDataset(
        args.dataset_path, split="test", transform=transforms
    )
    test_dataset = FmriDataset(
        args.dataset_path, split="train", transform=transforms
    )

    batch_size = args.batch_size
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=16,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=16,
    )

    return train_dataloader, test_dataloader


def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(
        torch.cat(
            [
                torch.cat([i for i in images.cpu()], dim=-1),
            ],
            dim=-2,
        )
        .permute(1, 2, 0)
        .cpu()
    )
    plt.show()


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to("cpu").numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)
