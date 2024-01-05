import numpy as np
from nilearn import masking, maskers
import nibabel as nib
from sklearn.decomposition import PCA, FastICA
from joblib import Memory
from pathlib import Path
from fugw.utils import load_mapping


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


def compute_pca(features, n_components=1024):
    """Compute PCA on features

    Parameters
    ----------
    features : np.ndarray
        Features to compute PCA on
    n_components : int, optional
        Number of components to keep, by default 1024

    Returns
    -------
    PCA
        PCA object
    """
    pca = PCA(n_components=n_components)
    pca.fit(features)
    return pca


def compute_transformed_features_pca(features, pca):
    """Compute transformed features

    Parameters
    ----------
    features : np.ndarray
        Features to transform
    pca : PCA
        PCA object

    Returns
    -------
    np.ndarray
        Transformed features
    """
    return pca.transform(features)


def save_pca(pca, transformed_features, output_folder):
    """Save PCA components

    Parameters
    ----------
    pca : PCA
        PCA object
    transformed_features : np.ndarray
        Transformed features
    output_folder : str
        Path to the output folder
    """
    if not output_folder.exists():
        output_folder.mkdir(parents=True)
    np.save(
        output_folder / "pca_components.npy",
        pca.components_,
    )
    # Save transformed features
    np.save(
        output_folder / "transformed_features.npy",
        transformed_features,
    )


def compute_ica(features, n_components=1024):
    """Compute ICA on features

    Parameters
    ----------
    features : np.ndarray
        Features to compute ICA on
    n_components : int, optional
        Number of components to keep, by default 1024

    Returns
    -------
    ICA
        FastICA object
    """
    ica = FastICA(n_components=n_components)
    ica.fit(features)
    return ica


def compute_transformed_features_ica(features, ica):
    """Compute transformed features

    Parameters
    ----------
    features : np.ndarray
        Features to transform
    ica : ICA
        FastICA object

    Returns
    -------
    np.ndarray
        Transformed features
    """
    return ica.transform(features)


def save_ica(ica, transformed_features, output_folder):
    """Save ICA components

    Parameters
    ----------
    ica : ICA
        FastICA object
    transformed_features : np.ndarray
        Transformed features
    output_folder : str
        Path to the output folder
    """
    if not output_folder.exists():
        output_folder.mkdir(parents=True)
    np.save(
        output_folder / "ica_components.npy",
        ica.components_,
    )
    # Save transformed features
    np.save(
        output_folder / "transformed_features.npy",
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
