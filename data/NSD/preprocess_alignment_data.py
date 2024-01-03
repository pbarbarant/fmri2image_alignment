# %%
from utils import load_metadata
import pandas as pd
from pathlib import Path
import nibabel as nib
from nilearn import image
from joblib import Parallel, delayed


metadata_path = Path(
    "/storage/store3/work/pbarbara/fmri2image_alignment/data/NSD/nsd_stim_info_merged.csv"
)
metadata = load_metadata(metadata_path)
N_images = len(metadata)


# Select images where shared1000 is True
metadata = metadata[metadata["shared1000"] == True]

# Select the indices of the shared1000 images
idx_shared1000 = metadata["subject1_rep0"].values.astype(int).tolist()
img_ids_shared1000 = metadata["nsdId"].values.astype(int).tolist()


def get_contratst(sub, it, idx):
    ses, run = divmod(idx, 750)
    ses += 1
    run -= 1
    assert run >= 0 and run < 750, f"Run {run} is not valid"
    img_id = img_ids_shared1000[it]
    if ses <= 30:
        path = Path(
            f"/data/parietal/store3/data/natural_scenes/3mm/{sub}/betas_session{ses:02d}.nii.gz"
        )
        assert path.exists(), f"Path {path} does not exist"
        # Open the image
        contrasts = nib.load(path)
        # Select the run
        contrast = image.index_img(contrasts, run)

    return contrast, img_id


def subject_wrapper(sub):
    stack = []
    for it, idx in enumerate(idx_shared1000[]):
        if it % 100 == 0:
            print(
                f"Processing image {it} out of {len(idx_shared1000)} for {sub}\n"
            )
        stack.append(get_contratst(sub, it, idx))
    # Sort the stack by image id
    stack = sorted(stack, key=lambda x: x[1])

    # Concatenate the images
    concat = image.concat_imgs([x[0] for x in stack])
    image_ids = [x[1] for x in stack]

    outpath = Path(
        f"/data/parietal/store3/work/pbarbara/fmri2image_alignment/data/NSD/alignment_data"
    )
    if not outpath.parent.exists():
        outpath.parent.mkdir(parents=True)

    # Save the concatenated image
    concat.to_filename(outpath / f"{sub}_shared1000.nii.gz")
    # Save the image ids
    pd.DataFrame(image_ids).to_csv(
        outpath / f"{sub}_shared1000.csv", index=False, header=False
    )


if __name__ == "__main__":
    subjects = [f"subj{i:02d}" for i in range(1, 9)]
    Parallel(n_jobs=8)(delayed(subject_wrapper)(sub) for sub in subjects)
