# %%
from ddpm import train_ddpm

import wandb


def launch():
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_conditional"
    args.epochs = 300
    args.batch_size = 32
    args.image_size = 64
    args.mri_dim = 2
    args.dataset_path = r"/data/parietal/store3/work/pbarbara/fmri2image_alignment/data/NSD/dataset_unaligned/dataset.csv"
    args.device = "cuda"
    args.lr = 3e-4
    train_ddpm(args)


if __name__ == "__main__":
    # Initialize WandB
    wandb.init(
        project="fmri2image_alignment",
        name="DDPM_conditional",
        # mode="disabled",
    )
    launch()
    wandb.finish()
