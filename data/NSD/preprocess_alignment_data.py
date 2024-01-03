# %%
from utils import load_metadata
import pandas as pd
from pathlib import Path
import nibabel as nib
from nilearn import image


metadata_path = Path(
    "/storage/store3/work/pbarbara/fmri2image_alignment/data/NSD/nsd_stim_info_merged.csv"
)
metadata = load_metadata(metadata_path)
N_images = len(metadata)


# Select images where shared1000 is True
metadata = metadata[metadata["shared1000"] == True]

# Filter out images where subject1_rep0 is greater than 30*750
metadata = metadata[metadata["subject1_rep0"] <= 22274]

# Select the indices of the shared1000 images
contrasts_id = metadata["subject1_rep0"].values.astype(int).tolist()
img_ids_shared1000 = metadata["nsdId"].values.astype(int).tolist()


def get_contrasts(sub, indices):
    path = Path(
        f"/data/parietal/store3/data/natural_scenes/curated_3mm/{sub}.nii.gz"
    )
    assert path.exists(), f"Path {path} does not exist"
    # Open the large contrasts file
    print(f"Loading {path}")
    contrasts = nib.load(path)
    print(f"Loading complete")
    print(f"Shape: {contrasts.shape}")
    # Select the contrasts
    print(f"Slicing...")
    contrasts = image.index_img(
        contrasts,
        [i - 1 for i in indices],
    )
    print(f"Slicing complete")

    return contrasts


def subject_wrapper(sub, indices, image_ids):
    contrasts = get_contrasts(sub, indices)

    outpath = Path(
        f"/data/parietal/store3/work/pbarbara/fmri2image_alignment/data/NSD/alignment_data"
    )
    if not outpath.exists():
        outpath.mkdir(parents=True)

    # Save the concatenated image
    contrasts.to_filename(outpath / f"{sub}_shared1000.nii.gz")
    # Save the image ids
    pd.DataFrame(image_ids).to_csv(
        outpath / f"{sub}_shared1000.csv", index=False, header=False
    )
    print(f"Subject {sub} complete")


if __name__ == "__main__":
    subjects = [f"sub-{i:02d}" for i in range(1, 9)]
    for sub in subjects:
        subject_wrapper(sub, contrasts_id, img_ids_shared1000)
