# %%
from pathlib import Path

import numpy as np
import pandas as pd
from utils import load_metadata

metadata_path = Path(
    "/storage/store3/work/pbarbara/fmri2image_alignment/data/NSD/nsd_stim_info_merged.csv"
)
metadata = load_metadata(metadata_path)
N_images = len(metadata)


# Select images where shared1000 is True
metadata = metadata[metadata["shared1000"] is True]

# Filter out images where subject1_rep0 is greater than 30*750
metadata = metadata[metadata["subject1_rep0"] <= 22274]

# Select the indices of the shared1000 images
contrasts_id = metadata["subject1_rep0"].values.astype(int).tolist()
img_ids_shared1000 = metadata["nsdId"].values.astype(int).tolist()


def get_contrasts(sub, indices):
    path = Path(
        f"/storage/store3/work/pbarbara/fmri2image_alignment/data/NSD/masked_subjects/{sub}.npy"
    )
    assert path.exists(), f"Path {path} does not exist"
    # Open the large contrasts file
    contrasts = np.load(path)
    # Select the contrasts
    offset = [i - 1 for i in indices]
    return contrasts[offset, :]


def subject_wrapper(sub, indices, image_ids):
    contrasts = get_contrasts(sub, indices)

    outpath = Path(
        "/data/parietal/store3/work/pbarbara/fmri2image_alignment/data/NSD/alignment_data"
    )
    if not outpath.exists():
        outpath.mkdir(parents=True)

    # Save the concatenated image
    np.save(outpath / f"{sub}_shared1000.npy", contrasts)
    # Save the image ids
    pd.DataFrame(image_ids).to_csv(
        outpath / f"{sub}_shared1000.csv", index=False, header=False
    )
    print(f"Subject {sub} complete")


if __name__ == "__main__":
    subjects = [f"sub-{i:02d}" for i in range(1, 9)]
    for sub in subjects:
        subject_wrapper(sub, contrasts_id, img_ids_shared1000)
