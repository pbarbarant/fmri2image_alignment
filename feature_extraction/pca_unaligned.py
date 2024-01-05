# %%
import numpy as np
from pathlib import Path
from utils import (
    compute_pca,
    compute_transformed_features_pca,
    save_pca,
    load_visual_mask,
)


if __name__ == "__main__":
    # Load visual mask
    masker = load_visual_mask()

    # Load features
    subjects = [f"sub-0{i}" for i in range(1, 9)]
    features = []
    for sub in subjects:
        source_features = masker.transform(
            f"/storage/store3/work/pbarbara/fmri2image_alignment/data/NSD/alignment_data/{sub}_shared1000.nii.gz"
        )
        features.append(source_features)

    features = np.concatenate(features, axis=0)
    print(f"Features shape: {features.shape}")

    # Compute PCA
    pca = compute_pca(features)

    # Fit transformed features
    transformed_features = compute_transformed_features_pca(features, pca)

    # Save PCA components
    output_folder = Path(
        f"/storage/store3/work/pbarbara/fmri2image_alignment/data/NSD/pca_unaligned"
    )
    save_pca(pca, transformed_features, output_folder)
