# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Paths to ICA and PCA features
ica_aligned_path = Path(
    "/data/parietal/store3/work/pbarbara/fmri2image_alignment/feature_extraction/features/ica_aligned/extracted_features.npy"
)
ica_unaligned_path = Path(
    "/data/parietal/store3/work/pbarbara/fmri2image_alignment/feature_extraction/features/ica_unaligned/extracted_features.npy"
)
pca_aligned_path = Path(
    "/data/parietal/store3/work/pbarbara/fmri2image_alignment/feature_extraction/features/pca_aligned/extracted_features.npy"
)
pca_unaligned_path = Path(
    "/data/parietal/store3/work/pbarbara/fmri2image_alignment/feature_extraction/features/pca_unaligned/extracted_features.npy"
)

ica_aligned = np.load(ica_aligned_path)
ica_unaligned = np.load(ica_unaligned_path)
pca_aligned = np.load(pca_aligned_path)
pca_unaligned = np.load(pca_unaligned_path)


subjects = np.concatenate([np.ones(902) * i for i in range(2, 9)])


fig, axes = plt.subplots(1, 2, figsize=(15, 7))
axes = axes.flatten()
fig.suptitle("First two components of ICA features")
axes[0].scatter(
    ica_unaligned[:, 0],
    ica_unaligned[:, 1],
    c=subjects,
    cmap="tab10",
    s=2,
    alpha=0.5,
    label="subject",
)
axes[0].set_xlim(-20, 20)
axes[0].set_ylim(-20, 20)
axes[0].set_title("Unaligned data")
scatter = axes[1].scatter(
    ica_aligned[:, 0],
    ica_aligned[:, 1],
    c=subjects,
    cmap="tab10",
    s=2,
    alpha=0.5,
)
handles, _ = scatter.legend_elements(prop="colors", alpha=0.8)
axes[1].legend(
    handles,
    [f"sub-0{i}" for i in range(2, 9)],
    loc="center right",
    bbox_to_anchor=(1.25, 0.5),
)
axes[1].set_title("Aligned data")
axes[1].set_xlim(-20, 20)
axes[1].set_ylim(-20, 20)

# Save figure
fig.savefig(
    "/data/parietal/store3/work/pbarbara/fmri2image_alignment/figures/ica_plot.png",
    bbox_inches="tight",
)


fig, axes = plt.subplots(1, 2, figsize=(15, 7))
axes = axes.flatten()
fig.suptitle("First two components of PCA features")
axes[0].scatter(
    pca_unaligned[:, 0],
    pca_unaligned[:, 1],
    c=subjects,
    cmap="tab10",
    s=2,
    alpha=0.5,
)
axes[0].set_title("Unaligned data")
axes[0].set_xlim(-1000, 1000)
axes[0].set_ylim(-1000, 1000)
scatter = axes[1].scatter(
    pca_aligned[:, 0],
    pca_aligned[:, 1],
    c=subjects,
    cmap="tab10",
    s=2,
    alpha=0.5,
)
handles, _ = scatter.legend_elements(prop="colors", alpha=0.8)
axes[1].legend(
    handles,
    [f"sub-0{i}" for i in range(2, 9)],
    loc="center right",
    bbox_to_anchor=(1.2, 0.5),
)
axes[1].set_title("Aligned data")
axes[1].set_xlim(-1000, 1000)
axes[1].set_ylim(-1000, 1000)
for ax in axes:
    ax.set_aspect("equal")
    ax.set_box_aspect(1)
plt.tight_layout()
plt.show()

# Save figure
fig.savefig(
    "/data/parietal/store3/work/pbarbara/fmri2image_alignment/figures/pca_plot.png",
    bbox_inches="tight",
)
