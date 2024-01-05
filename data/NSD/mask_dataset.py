import numpy as np
import nibabel as nib
from nilearn import image, maskers, masking
from pathlib import Path
from joblib import Memory


def load_subject_NSD(subject_name, verbose=True):
    """Load subject from NSD dataset

    Parameters
    ----------
    subject : str
        Subject name

    Returns
    -------
    NiftiImage
        Subject image
    """
    # Load subject image
    subject_path = Path(
        f"/storage/store3/data/natural_scenes/curated_3mm/{subject_name}.nii.gz"
    )
    if verbose:
        print(f"Loading {subject_path}")
    subject_img = nib.load(subject_path)
    if verbose:
        print(f"Loading complete")

    return subject_img


def load_and_save_visual_mask():
    """Load visual mask

    Returns
    -------
    NiftiMasker
    """
    mni152_mask = nib.load(
        "/storage/store2/work/tbazeill/cneuromod_wm_5mm/gm_visual_mask.nii.gz"
    )
    connected_mask = masking.compute_background_mask(
        mni152_mask, connected=True
    )
    # Save mask
    mask_path = Path(
        "/storage/store3/work/pbarbara/fmri2image_alignment/data/NSD/masked_subjects/visual_mask.nii.gz"
    )
    nib.save(connected_mask, mask_path)

    # Create masker
    masker = maskers.NiftiMasker(connected_mask, memory=Memory()).fit()
    return masker


def apply_mask(subject_img, masker):
    """Apply mask to subject image

    Parameters
    ----------
    subject_img : NiftiImage
        Subject image
    masker : NiftiMasker
        Masker

    Returns
    -------
    np.ndarray
        Masked subject image
    """
    return masker.transform(subject_img)


def save_masked_subject(subject_name, masked_subject, verbose=True):
    """Save masked subject

    Parameters
    ----------
    subject : str
        Subject name
    masked_subject : np.ndarray
        Masked subject image
    """
    # Save masked subject
    masked_subject_path = Path(
        f"/storage/store3/work/pbarbara/fmri2image_alignment/data/NSD/masked_subjects/{subject_name}.npy"
    )

    if verbose:
        print(f"Saving {masked_subject_path}")
    np.save(masked_subject_path, masked_subject)
    if verbose:
        print(f"Saving complete")


if __name__ == "__main__":
    subjects = [f"sub-{i:02d}" for i in range(1, 9)]
    masker = load_and_save_visual_mask()
    for subject in subjects:
        print(f"Processing {subject}")
        subject_img = load_subject_NSD(subject)
        masked_subject = apply_mask(subject_img, masker)
        save_masked_subject(subject, masked_subject)
