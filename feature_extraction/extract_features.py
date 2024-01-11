# %%
from pathlib import Path

import numpy as np

from utils import (
    compute_transformed_features_ica,
    compute_transformed_features_pca,
    load_mapping_from_path,
    project_on_target,
    save_features,
)

if __name__ == "__main__":
    # Load features
    target = "sub-01"
    sources = [f"sub-0{i}" for i in range(2, 9)]
    path_to_mapping_folder = Path(
        "/data/parietal/store3/work/pbarbara/fmri2image_alignment/alignment/mappings"
    )
    assert path_to_mapping_folder.exists(), "Mappings folder does not exist"

    features_aligned = []
    features_unaligned = []
    for source in sources:
        print(f"Processing {source}")
        mapping = load_mapping_from_path(
            source, target, path_to_mapping_folder
        )
        source_features = np.load(
            f"/storage/store3/work/pbarbara/fmri2image_alignment/data/NSD/alignment_data/{source}_shared1000.npy"
        )
        projected_source = project_on_target(source_features, mapping)

        # Append features
        features_unaligned.append(source_features)
        features_aligned.append(projected_source)

    stack_features_unaligned = np.concatenate(features_aligned, axis=0)
    stack_features_aligned = np.concatenate(features_unaligned, axis=0)

    # Load target features
    target_features = np.load(
        f"/storage/store3/work/pbarbara/fmri2image_alignment/data/NSD/alignment_data/{target}_shared1000.npy"
    )

    # Fit transformed features
    (
        transformed_features_unaligned_pca_train,
        transformed_features_unaligned_pca_test,
        pca_unaligned,
    ) = compute_transformed_features_pca(
        stack_features_unaligned,
        target_features,
    )
    print("Unaligned PCA done")
    (
        transformed_features_aligned_pca_train,
        transformed_features_aligned_pca_test,
        pca_aligned,
    ) = compute_transformed_features_pca(
        stack_features_aligned,
        target_features,
    )
    print("Aligned PCA done")
    (
        transformed_features_unaligned_ica_train,
        transformed_features_unaligned_ica_test,
        ica_unaligned,
    ) = compute_transformed_features_ica(
        stack_features_unaligned,
        target_features,
    )
    print("Unaligned ICA done")
    (
        transformed_features_aligned_ica_train,
        transformed_features_aligned_ica_test,
        ica_aligned,
    ) = compute_transformed_features_ica(
        stack_features_aligned,
        target_features,
    )
    print("Aligned ICA done")

    # Save PCA components
    output_folder = Path(
        "/storage/store3/work/pbarbara/fmri2image_alignment/feature_extraction/features/"
    )
    save_features(
        transformed_features_unaligned_pca_train,
        transformed_features_unaligned_pca_test,
        pca_unaligned,
        output_folder / "pca_unaligned",
    )
    save_features(
        transformed_features_aligned_pca_train,
        transformed_features_aligned_pca_test,
        pca_aligned,
        output_folder / "pca_aligned",
    )
    save_features(
        transformed_features_unaligned_ica_train,
        transformed_features_unaligned_ica_test,
        ica_unaligned,
        output_folder / "ica_unaligned",
    )
    save_features(
        transformed_features_aligned_ica_train,
        transformed_features_aligned_ica_test,
        ica_aligned,
        output_folder / "ica_aligned",
    )
