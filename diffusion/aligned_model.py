from ddpm_conditional import train

import wandb


def launch():
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_conditional_aligned"
    args.epochs = 600
    args.freeze_after_epoch = 300
    args.batch_size = 32
    args.image_size = 64
    args.mri_dim = 19450
    args.dataset_path = r"/data/parietal/store3/work/pbarbara/fmri2image_alignment/data/NSD/dataset_aligned/dataset.csv"
    args.device = "cuda"
    args.lr = 3e-4
    train(args)


if __name__ == "__main__":
    # Initialize WandB
    wandb.init(
        project="fmri2image_alignment",
        name="diffusion_aligned",
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
