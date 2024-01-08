import numpy as np
import pandas as pd
import pickle as pkl
from pathlib import Path


def load_features_from_model(model="pca_unaligned"):
    """Load features from feature extraction

    Parameters
    ----------
    model : str, optional
        Saved model, by default "pca_unaligned"

    Returns
    -------
    ndarray
        Features
    """
    X = np.load(
        f"/storage/store3/work/pbarbara/fmri2image_alignment/feature_extraction/features/{model}/extracted_features.npy"
    )
    return X


def load_labels_share1000():
    """Load supercategories labels for shared1000 images

    Returns
    -------
    ndarray
        Object labels
    """
    # Path to coco annotations
    annotations_path = Path(
        "/storage/store3/data/natural_scenes/info/coco_annotations/coco_categories.pkl"
    )

    metadata_path = Path(
        "/storage/store3/work/pbarbara/fmri2image_alignment/data/NSD/nsd_stim_info_merged.csv"
    )

    # Load annotations
    with open(annotations_path, "rb") as f:
        annotations = pkl.load(f)

    annotations = pd.DataFrame(annotations)
    annotations = annotations[["id", "supercat"]]
    annotations["id"] = annotations["id"].astype(int) - 1

    # Load metadata
    metadata = pd.read_csv(metadata_path)

    # Merge annotations and metadata on nsdId
    metadata = metadata.merge(annotations, left_on="nsdId", right_on="id")
    metadata = metadata[
        (metadata["shared1000"] is True) & (metadata["subject1_rep0"] <= 22274)
    ]

    metadata = metadata[["nsdId", "supercat"]]

    labels = np.concatenate([np.array(metadata["supercat"])] * 7).flatten()
    return labels
