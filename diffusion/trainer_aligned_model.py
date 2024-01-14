# %%
from ddpm import train_embeddings

import wandb


def launch():
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "aligned_embeddings"
    args.epochs = 300
    args.batch_size = 32
    args.image_size = 64
    args.mri_dim = 2
    args.model_path = r"/data/parietal/store3/work/pbarbara/fmri2image_alignment/models/DDPM_conditional/ckpt.pt"
    args.dataset_path = r"/data/parietal/store3/work/pbarbara/fmri2image_alignment/data/NSD/dataset_aligned/dataset.csv"
    args.device = "cuda"
    args.lr = 3e-4
    train_embeddings(args)


if __name__ == "__main__":
    # Initialize WandB
    wandb.init(
        project="fmri2image_alignment",
        name="aligned_embeddings",
        # mode="disabled",
    )
    launch()
    wandb.finish()
    # device = "cuda"
    # model = UNet_conditional(mri_dim=2).to(device)
    # model = nn.DataParallel(model)
    # ckpt = torch.load(
    #     "/data/parietal/store3/work/pbarbara/fmri2image_alignment/models/DDPM_conditional_aligned/ckpt.pt"
    # )
    # model.load_state_dict(ckpt)
    # diffusion = Diffusion(img_size=64, device=device)
    # n = 8
    # y = torch.Tensor([6] * n).long().to(device)
    # x = diffusion.sample(model, n, None)
    # plot_images(x)
