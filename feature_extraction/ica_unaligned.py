# %%
import numpy as np
from pathlib import Path
from utils import (
    compute_ica,
    compute_transformed_features_ica,
    save_ica,
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

    # Compute ICA
    ica = compute_ica(features)

    # Fit transformed features
    transformed_features = compute_transformed_features_ica(features, ica)

    # Save ICA components
    output_folder = Path(
        f"/storage/store3/work/pbarbara/fmri2image_alignment/data/NSD/ica_unaligned"
    )
    save_ica(ica, transformed_features, output_folder)
