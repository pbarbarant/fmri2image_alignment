# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import umap
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":
    # Load features
    target = "sub-01"
    sources = [f"sub-0{i}" for i in range(2, 9)]
    se = StandardScaler()

    features_aligned = []
    features_unaligned = []
    for source in sources:
        print(f"Processing {source}")
        source_features = np.load(
            f"/data/parietal/store3/work/pbarbara/fmri2image_alignment/data/NSD/masked_subjects/{source}.npy"
        )

        projected_source = np.load(
            f"/data/parietal/store3/work/pbarbara/fmri2image_alignment/data/NSD/projected_features/{source}_{target}.npy"
        )

        source_features = se.fit_transform(source_features)
        projected_source = se.fit_transform(projected_source)

        # Append features
        features_unaligned.append(source_features)
        features_aligned.append(projected_source)

    stack_features_unaligned = np.concatenate(features_unaligned, axis=0)
    stack_features_aligned = np.concatenate(features_aligned, axis=0)

    # Load target features
    target_features = np.load(
        f"/data/parietal/store3/work/pbarbara/fmri2image_alignment/data/NSD/masked_subjects/{target}.npy"
    )
    target_features = se.fit_transform(target_features)

    # Append target features
    stack_features_unaligned = np.concatenate(
        [target_features, stack_features_unaligned], axis=0
    )
    stack_features_aligned = np.concatenate(
        [target_features, stack_features_aligned], axis=0
    )

    labels = np.concatenate(
        [i * np.ones(len(target_features)) for i in range(1, 9)]
    )
    print(labels.shape)

    # Use UMAP
    reducer = umap.UMAP(verbose=True)
    reducer.fit(stack_features_unaligned)
    transformed_features_unaligned_umap = reducer.transform(
        stack_features_unaligned
    )
    print("Unaligned UMAP done")
    reducer.fit(stack_features_aligned)
    transformed_features_aligned_umap = reducer.transform(
        stack_features_aligned
    )
    print("Aligned UMAP done")

    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    ax[0].scatter(
        transformed_features_unaligned_umap[:, 0],
        transformed_features_unaligned_umap[:, 1],
        c=labels,
        cmap="tab10",
        s=2,
        alpha=0.5,
    )
    ax[0].set_title("Unaligned data")
    ax[0].set_aspect("equal", "datalim")

    scatter = ax[1].scatter(
        transformed_features_aligned_umap[:, 0],
        transformed_features_aligned_umap[:, 1],
        c=labels,
        cmap="tab10",
        s=2,
        alpha=0.5,
    )
    ax[1].set_title("Aligned data")
    handles, _ = scatter.legend_elements(prop="colors", alpha=0.5)
    ax[1].legend(
        handles,
        [f"sub-0{i}" for i in range(1, 9)],
        loc="center right",
        bbox_to_anchor=(1.2, 0.5),
    )
    ax[1].set_aspect("equal", "datalim")

    fig.suptitle("UMAP projection of NSD features", fontsize=16)
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    plt.savefig(
        "/data/parietal/store3/work/pbarbara/fmri2image_alignment/figures/umap_projection.png",
        bbox_inches="tight",
    )
    plt.show()

    # Save the components
    saving_folder = Path(
        "/data/parietal/store3/work/pbarbara/fmri2image_alignment/feature_extraction/umap_components"
    )
    if not saving_folder.exists():
        saving_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

    print("Saving components")
    np.save(
        saving_folder / "umap_components_unaligned.npy",
        transformed_features_unaligned_umap,
    )

    np.save(
        saving_folder / "umap_components_aligned.npy",
        transformed_features_aligned_umap,
    )
