# %%
import numpy as np
from pathlib import Path
import umap
import matplotlib.pyplot as plt


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
        source_features = np.load(
            f"/storage/store3/work/pbarbara/fmri2image_alignment/data/NSD/masked_subjects/{source}.npy"
        )

        projected_source = np.load(
            f"/storage/store3/work/pbarbara/fmri2image_alignment/data/NSD/projected_features/{source}_{target}.npy"
        )

        # Append features
        features_unaligned.append(source_features)
        features_aligned.append(projected_source)

    stack_features_unaligned = np.concatenate(features_aligned, axis=0)
    stack_features_aligned = np.concatenate(features_unaligned, axis=0)

    # Load target features
    target_features = np.load(
        f"/storage/store3/work/pbarbara/fmri2image_alignment/data/NSD/alignment_data/{target}_shared1000.npy"
    )

    labels = np.concatenate(
        [i * np.ones(len(source_features)) for i in range(2, 9)]
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
    )
    ax[0].set_title("Unaligned data")
    ax[0].set_aspect("equal", "datalim")

    ax[1].scatter(
        transformed_features_aligned_umap[:, 0],
        transformed_features_aligned_umap[:, 1],
        c=labels,
        cmap="tab10",
    )
    ax[1].set_title("Aligned data")

    # Add legend to colors
    # Load handles from tab10 colormap
    handles = [
        plt.scatter(
            [None, None],
            [None, None],
            color=c,
        )
        for c in plt.cm.tab10(np.arange(10))
    ]
    ax[1].legend(
        handles,
        [f"sub-0{i}" for i in range(2, 9)],
        loc="center right",
        bbox_to_anchor=(1.2, 0.5),
    )
    ax[1].set_aspect("equal", "datalim")

    fig.suptitle("UMAP projection of NSD features", fontsize=16)
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    plt.savefig(
        "/storage/store3/work/pbarbara/fmri2image_alignment/figures/umap_projection.png",
        bbox_inches="tight",
    )
    plt.show()
