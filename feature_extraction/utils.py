from pathlib import Path

import nibabel as nib
import numpy as np
from fugw.utils import load_mapping
from joblib import Memory
from nilearn import maskers, masking
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler


def load_mapping_from_path(source, target, path):
    """Load mapping from file

    Parameters
    ----------
    path : str
        Path to the mapping folder

    Returns
    -------
    np.ndarray
        Mapping
    """
    mapping_path = Path(path) / f"{source}_{target}.pkl"
    mapping = load_mapping(mapping_path)
    return mapping


def project_on_target(source_features, mapping):
    """Project source image onto target image

    Parameters
    ----------
    source : np.ndarray
        Source features
    mapping : np.ndarray
        Mapping from source to target

    Returns
    -------
    NiftiImage
        Projected source image
    """
    predicted_target_features = mapping.transform(source_features)

    return predicted_target_features


def compute_transformed_features_pca(features, n_components=1024):
    """Compute transformed features

    Parameters
    ----------
    features : np.ndarray
        Features to transform
    n_components : int, optional
        Number of components to keep, by default 1024


    Returns
    -------
    np.ndarray
        Transformed features
    """
    # Standardize features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    # Compute PCA
    pca = PCA(n_components=n_components)
    return pca.fit_transform(features)


def compute_transformed_features_ica(features, n_components=1024):
    """Compute transformed features

    Parameters
    ----------
    features : np.ndarray
        Features to transform
    n_components : int, optional
        Number of components to keep, by default 1024


    Returns
    -------
    np.ndarray
        Transformed features
    """
    # Standardize features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    # Compute ICA
    ica = FastICA(n_components=n_components)
    return ica.fit_transform(features)


def save_features(transformed_features, output_folder):
    """Save PCA components

    Parameters
    ----------
    transformed_features : np.ndarray
        Transformed features
    output_folder : str
        Path to the output folder
    """
    if not output_folder.exists():
        output_folder.mkdir(parents=True)
    # Save transformed features
    np.save(
        output_folder / "extracted_features.npy",
        transformed_features,
    )


def load_visual_mask():
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
    masker = maskers.NiftiMasker(connected_mask, memory=Memory()).fit()
    return masker