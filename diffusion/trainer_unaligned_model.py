# %%
from ddpm import train_embeddings

import wandb


def launch():
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "unaligned_embeddings"
    args.epochs = 300
    args.batch_size = 32
    args.image_size = 64
    args.mri_dim = 2
    args.model_path = r"/data/parietal/store3/work/pbarbara/fmri2image_alignment/models/DDPM_conditional/ckpt.pt"
    args.dataset_path = r"/data/parietal/store3/work/pbarbara/fmri2image_alignment/data/NSD/dataset_unaligned/dataset.csv"
    args.device = "cuda"
    args.lr = 3e-4
    train_embeddings(args)


if __name__ == "__main__":
    # Initialize WandB
    wandb.init(
        project="fmri2image_alignment",
        name="unaligned_embeddings",
        # mode="disabled",
    )
    launch()
    wandb.finish()
