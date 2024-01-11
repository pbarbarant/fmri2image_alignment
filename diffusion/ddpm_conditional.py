# %%

import os
import wandb
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from utils import plot_images, save_images, setup_logging, get_data
from modules import UNet_conditional
import logging

from accelerate import Accelerator


logging.basicConfig(
    format="%(asctime)s - %(levelname)s: %(message)s",
    level=logging.INFO,
    datefmt="%I:%M:%S",
)


class Diffusion:
    def __init__(
        self,
        noise_steps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        img_size=64,
        device="cuda",
    ):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.device = device

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[
            :, None, None, None
        ]
        eps = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * eps, eps

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n, y):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(
                self.device
            )
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, y)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = (
                    1
                    / torch.sqrt(alpha)
                    * (
                        x
                        - ((1 - alpha) / (torch.sqrt(1 - alpha_hat)))
                        * predicted_noise
                    )
                    + torch.sqrt(beta) * noise
                )
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x


def train(args):
    setup_logging(args.run_name)

    accelerator = Accelerator(mixed_precision="fp16")
    device = accelerator.device

    train_dataloader, test_dataloader = get_data(args)
    model = UNet_conditional(mri_dim=args.mri_dim)
    model = nn.DataParallel(model)
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    length = len(train_dataloader)

    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(train_dataloader)
        for i, batch in enumerate(pbar):
            optimizer.zero_grad()
            images = batch["image"]
            fmris = batch["fmri"]
            images = images.to(device)
            fmris = fmris.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            predicted_noise = model(x_t, t, fmris)
            loss = mse(noise, predicted_noise)

            accelerator.backward(loss)
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())
            wandb.log({"MSE": loss.item()}, step=epoch * length + i)

        if epoch % 10 == 0:
            fmris = next(iter(test_dataloader))["fmri"][:5].to(device)
            sampled_images = diffusion.sample(model, n=5, y=fmris)
            plot_images(sampled_images)
            save_images(
                sampled_images,
                os.path.join("results", args.run_name, f"{epoch}.jpg"),
            )
            torch.save(
                model.state_dict(),
                os.path.join("models", args.run_name, "ckpt.pt"),
            )
            torch.save(
                optimizer.state_dict(),
                os.path.join("models", args.run_name, "optim.pt"),
            )
            wandb.log(
                {"Sampled_Images": [wandb.Image(sampled_images[0])]},
                step=epoch,
            )


def launch():
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_conditional_unaligned"
    args.epochs = 300
    args.batch_size = 32
    args.image_size = 64
    args.mri_dim = 19450
    args.dataset_path = r"/data/parietal/store3/work/pbarbara/fmri2image_alignment/data/NSD/dataset_unaligned/dataset.csv"
    args.device = "cuda"
    args.lr = 3e-4
    train(args)


if __name__ == "__main__":
    # Initialize WandB
    wandb.init(
        project="fmri2image_alignment",
        name="diffusion_unaligned",
        # mode="disabled",
    )
    launch()
    wandb.finish()
    # device = "cuda"
    # model = UNet_conditional(mri_dim=10).to(device)
    # ckpt = torch.load("./models/DDPM_conditional/ckpt.pt")
    # model.load_state_dict(ckpt)
    # diffusion = Diffusion(img_size=64, device=device)
    # n = 8
    # y = torch.Tensor([6] * n).long().to(device)
    # x = diffusion.sample(model, n, y, cfg_scale=0)
    # plot_images(x)
