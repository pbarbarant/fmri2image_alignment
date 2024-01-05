# %%
import numpy as np
from pathlib import Path
from utils import (
    compute_pca,
    compute_transformed_features_pca,
    save_pca,
    load_mapping_from_path,
    project_on_target,
    load_visual_mask,
)


if __name__ == "__main__":
    # Load visual mask
    masker = load_visual_mask()

    # Load features
    target = "sub-01"
    sources = [f"sub-0{i}" for i in range(2, 9)]
    path_to_mapping_folder = Path(
        "/data/parietal/store3/work/pbarbara/fmri2image_alignment/data/NSD/mappings"
    )
    assert path_to_mapping_folder.exists(), "Mappings folder does not exist"

    features = [
        masker.transform(
            f"/storage/store3/work/pbarbara/fmri2image_alignment/data/NSD/alignment_data/{target}_shared1000.nii.gz"
        )
    ]
    for source in sources:
        print(f"Processing {source}")
        mapping = load_mapping_from_path(
            source, target, path_to_mapping_folder
        )
        source_features = masker.transform(
            f"/storage/store3/work/pbarbara/fmri2image_alignment/data/NSD/alignment_data/{source}_shared1000.nii.gz"
        )
        projected_source = project_on_target(source_features, mapping)
        features.append(projected_source)

    features = np.concatenate(features, axis=0)
    print(f"Features shape: {features.shape}")

    # Compute PCA
    pca = compute_pca(features)

    # Fit transformed features
    transformed_features = compute_transformed_features_pca(features, pca)

    # Save PCA components
    output_folder = Path(
        f"/storage/store3/work/pbarbara/fmri2image_alignment/data/NSD/pca_{target}"
    )
    save_pca(pca, transformed_features, output_folder)
