# %%
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from ddpm import Diffusion
from ignite.metrics import SSIM
from modules import UNet_conditional
from PIL import Image
from scipy.linalg import sqrtm
from torchvision import transforms
from torchvision.models import inception_v3


def compute_ssim(y_pred_normalized, y_true_normalized):
    ssim_values = []
    for i in range(y_pred_normalized.shape[0]):
        ssim_metric = SSIM(data_range=1.0)
        ssim_metric.update(
            (
                y_pred_normalized[i, ...].unsqueeze(0),
                y_true_normalized[i, ...].unsqueeze(0),
            )
        )
        ssim_value = ssim_metric.compute()
        ssim_values.append(ssim_value)
        # Reset the metric
        ssim_metric.reset()
    return ssim_values


def calculate_fid_score(real_features, generated_features):
    mu_real, sigma_real = torch.mean(real_features, dim=0), torch_cov(
        real_features, rowvar=False
    )
    mu_generated, sigma_generated = torch.mean(
        generated_features, dim=0
    ), torch_cov(generated_features, rowvar=False)

    # Calculate Frechet distance
    diff = mu_real - mu_generated
    covmean, _ = sqrtm(sigma_real @ sigma_generated, disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma_real.shape[0]) * 1e-6
        covmean = sqrtm((sigma_real + offset) @ (sigma_generated + offset))

    fid_score = (
        diff.dot(diff)
        + torch.trace(sigma_real)
        + torch.trace(sigma_generated)
        - 2 * torch.trace(torch.from_numpy(covmean))
    ).real

    return fid_score.item()


def torch_cov(m, rowvar=False):
    """
    Estimate covariance matrix of m.
    """
    if m.dim() > 2:
        raise ValueError("m has more than 2 dimensions")
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    m -= torch.mean(m, dim=1, keepdim=True)
    factor = 1 / (m.size(1) - 1)
    return factor * m.matmul(m.t())


def compute_fid_list(y_pred_normalized, y_true_normalized):
    # Resize the tensors to a larger size for Inception model
    resize_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
        ]
    )
    y_pred_resized = torch.stack(
        [resize_transform(img) for img in y_pred_normalized]
    )
    y_true_resized = torch.stack(
        [resize_transform(img) for img in y_true_normalized]
    )

    # Load InceptionV3 model
    inception_model = inception_v3(
        pretrained=True,
        transform_input=False,
    )
    inception_model.eval()

    fid_scores = []
    for i in range(y_pred.shape[0]):
        # Extract features from InceptionV3
        with torch.no_grad():
            real_features = torch.tensor(
                inception_model(y_true_resized[i].unsqueeze(0))[0]
                .squeeze()
                .cpu()
                .numpy()
                .reshape(1, -1)
            )
            generated_features = torch.tensor(
                inception_model(y_pred_resized[i].unsqueeze(0))[0]
                .squeeze()
                .cpu()
                .numpy()
                .reshape(1, -1)
            )

        fid_score = calculate_fid_score(
            real_features,
            generated_features,
        )
        fid_scores.append(fid_score)
    return fid_scores


def get_fmri(fmri_path):
    fmri = np.load(fmri_path)
    return fmri


def get_image(image_path):
    image = Image.open(image_path)
    return image


def get_paths(subject):
    metadata = pd.read_csv(
        "/data/parietal/store3/work/pbarbara/fmri2image_alignment/data/NSD/dataset_aligned/dataset.csv"
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
    ckpt_path = "/data/parietal/store3/work/pbarbara/fmri2image_alignment/models/aligned_embeddings/ckpt2.pt"
    model = load_model(ckpt_path, device)
    diffusion = Diffusion(img_size=64, device=device)
    N = 1
    fmri_all_subjects, image_paths = retrieve_all_subjects(N=N)
    print("Successfully retrieved all subjects test data")

    # Preprocess the image
    image = get_image(image_paths[0])
    image = image.resize((64, 64))
    image = np.array(image)
    image = np.repeat(image[np.newaxis, ...], 8, axis=0)

    y_true = torch.Tensor(np.array(image)).to(device).float()
    # Permutes the dimensions to match the diffusion model
    y_true = y_true.permute(0, 3, 1, 2)
    y_pred = diffusion.sample(model, 8, fmri_all_subjects[:, 0, ...]).float()

    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    y_pred_normalized = normalize(y_pred)
    y_true_normalized = normalize(y_true)

    fid_list = compute_fid_list(y_pred_normalized, y_true_normalized)
    ssim_list = compute_ssim(y_pred_normalized, y_true_normalized)

    # Put in a dataframe
    df = pd.DataFrame(
        {
            "fid": fid_list,
            "ssim": ssim_list,
        }
    )
    output_folder = Path(
        "/data/parietal/store3/work/pbarbara/fmri2image_alignment/metrics/aligned_embeddings"
    )
    if not output_folder.exists():
        output_folder.mkdir(parents=True)
    df.to_csv(output_folder / "metrics.csv", index=False)
