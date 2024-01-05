# %%
import os
from pathlib import Path
from utils import FugwAlignment
from nilearn import maskers, masking, datasets, image
from fugw.utils import load_mapping, save_mapping
import nibabel as nib
import numpy as np
from joblib import Memory


def compute_pairwise_mapping(source, target, masker, mapping_path):
    """Align two images using FUGW

    Parameters
    ----------
    source : ndarray
        Source image
    target : ndarray
        Target image

    Returns
    -------
    None
    """
    # Initialize FUGW alignment
    alignment = FugwAlignment(
        masker=masker,
        method="coarse_to_fine",
        n_samples=1000,
        alpha_coarse=0.1,
        rho_coarse=1,
        eps_coarse=1e-6,
        alpha_fine=0.1,
        rho_fine=1,
        eps_fine=1e-6,
        radius=15,
    )

    # Fit alignment
    alignment.fit(source, target)

    # Save mapping
    save_mapping(alignment.mapping, mapping_path)


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
    ndarray
        Loaded image
    """

    sub_path = Path(
        f"/storage/store3/work/pbarbara/fmri2image_alignment/data/NSD/alignment_data/{subject}_shared1000.npy"
    )

    # Load volume
    features = np.load(sub_path)

    if verbose:
        print(f"Successfully loaded {sub_path}")

    return features


if __name__ == "__main__":
    # Define sources and target
    sources = [f"sub-{i:02d}" for i in range(2, 9)]
    target = "sub-01"
    target_features = load_subject_NSD(target)

    # Load visual mask
    mni152_mask = nib.load(
        "/storage/store2/work/tbazeill/cneuromod_wm_5mm/gm_visual_mask.nii.gz"
    )
    connected_mask = masking.compute_background_mask(
        mni152_mask, connected=True
    )
    masker = maskers.NiftiMasker(connected_mask, memory=Memory()).fit()

    for source in sources:
        print(f"Aligning {source} on {target}")

        source_features = load_subject_NSD(source)

        saving_dir = Path(
            f"/storage/store3/work/pbarbara/fmri2image_alignment/alignment/mappings/"
        )
        if not saving_dir.exists():
            os.makedirs(saving_dir)

        mapping_path = saving_dir / f"{source}_{target}.pkl"

        compute_pairwise_mapping(
            source_features, target_features, masker, mapping_path
        )
