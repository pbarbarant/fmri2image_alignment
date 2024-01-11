from pathlib import Path

import numpy as np
from fugw.utils import load_mapping
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler
from joblib import dump


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


def compute_transformed_features_pca(
    train_features, test_features, n_components=1024
):
    """Compute transformed features

    Parameters
    ----------
    train_features : np.ndarray
        Features to fit the ica
    test_features : np.ndarray
        Target features to transform
    n_components : int, optional
        Number of components to keep, by default 1024


    Returns
    -------
    np.ndarray
        Transformed features
    """
    # Standardize features
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    # Compute PCA
    pca = PCA(n_components=n_components)
    # Scale target features
    test_features = scaler.transform(test_features)

    transformed_features_train = pca.fit_transform(train_features)
    transformed_features_test = pca.transform(test_features)

    return (
        transformed_features_train,
        transformed_features_test,
        pca,
    )


def compute_transformed_features_ica(
    train_features, test_features, n_components=1024
):
    """Compute transformed features

    Parameters
    ----------
    train_features : np.ndarray
        Features to fit the ica
    test_features : np.ndarray
        Target features to transform
    n_components : int, optional
        Number of components to keep, by default 1024


    Returns
    -------
    np.ndarray
        Transformed features
    """
    # Standardize features
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    # Compute ICA
    ica = FastICA(n_components=n_components)
    # Scale target features
    test_features = scaler.transform(test_features)

    transformed_features_train = ica.fit_transform(train_features)
    transformed_features_test = ica.transform(test_features)

    return (
        transformed_features_train,
        transformed_features_test,
        ica,
    )


def save_features(
    transformed_features_train, transformed_features_test, model, output_folder
):
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
        output_folder / "extracted_features_train.npy",
        transformed_features_train,
    )
    np.save(
        output_folder / "extracted_features_test.npy",
        transformed_features_test,
    )
    # Save model
    dump(model, output_folder / "model.joblib")
