# %%
import os
from pathlib import Path
from utils import FugwAlignment
from nilearn import maskers, masking, datasets, plotting
import nibabel as nib
import numpy as np

# Download MNI152 template
mni152_mask = datasets.fetch_icbm152_brain_gm_mask()
connected_mask = masking.compute_background_mask(mni152_mask, connected=True)
masker = maskers.NiftiMasker(connected_mask).fit()


def compute_pairwise_mapping(source, target, masker, mapping_path):
    """Align two images using FUGW

    Parameters
    ----------
    source : NiftiImage
        Source image
    target : NiftiImage
        Target image

    Returns
    -------
    NiftiImage
        Aligned source image
    """
    # Initialize FUGW alignment
    alignment = FugwAlignment(masker=masker)

    # Fit alignment
    alignment.fit(source, target)

    # Save mapping
    alignment.save_mapping(mapping_path)


def load_subject_NSD(subject, verbose=True):
    """Load subject from NSD dataset

    Parameters
    ----------
    subject : str
        Subject name
    verbose : bool, optional
        Whether to print information about the loaded image, by default True

    Returns
    -------
    NiftiImage
        Loaded image
    """

    sub_path = Path(
        f"/storage/store3/work/pbarbara/fmri2image_alignment/data/NSD/alignment_data/{subject}_shared1000.nii.gz"
    )

    # Load volume
    volume = nib.load(sub_path)

    if verbose:
        print(f"Successfully loaded {sub_path}")

    return volume


if __name__ == "__main__":
    sources = [f"sub-{i:02d}" for i in range(2, 9)]
    target = "sub-01"

    for source in sources:
        print(f"Aligning {source} on {target}")

        source_img = load_subject_NSD(source)
        target_img = load_subject_NSD(target)

        saving_dir = Path(
            f"/storage/store3/work/pbarbara/fmri2image_alignment/data/NSD/mappings/"
        )
        if not saving_dir.exists():
            os.makedirs(saving_dir)

        mapping_path = saving_dir / f"{source}_{target}.pkl"

        compute_pairwise_mapping(source_img, target_img, masker, saving_dir)
