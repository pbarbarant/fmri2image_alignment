# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle as pkl
from sklearn.preprocessing import LabelEncoder


# Paths to pca and ica features
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


fig, axes = plt.subplots(1, 4, figsize=(12, 4))
axes[0].scatter(
    ica_aligned[:, 0],
    ica_aligned[:, 1],
    c=subjects,
    cmap="tab10",
    label="subject",
)
axes[0].set_xlim(-20, 20)
axes[0].set_ylim(-20, 20)
axes[0].set_title("ICA aligned")
axes[1].scatter(
    ica_unaligned[:, 0], ica_unaligned[:, 1], c=subjects, cmap="tab10"
)
axes[1].set_title("ICA unaligned")
axes[1].set_xlim(-20, 20)
axes[1].set_ylim(-20, 20)
axes[2].scatter(pca_aligned[:, 0], pca_aligned[:, 1], c=subjects, cmap="tab10")
axes[2].set_title("PCA aligned")
axes[2].set_xlim(-1000, 1000)
axes[2].set_ylim(-1000, 1000)
axes[3].scatter(
    pca_unaligned[:, 0], pca_unaligned[:, 1], c=subjects, cmap="tab10"
)
axes[3].set_title("PCA unaligned")
axes[3].set_xlim(-1000, 1000)
axes[3].set_ylim(-1000, 1000)
for ax in axes:
    ax.set_aspect("equal")
    ax.set_box_aspect(1)
plt.tight_layout()
plt.show()


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
    (metadata["shared1000"] == True) & (metadata["subject1_rep0"] <= 22274)
]

metadata = metadata[["nsdId", "supercat"]]

labels = np.concatenate([np.array(metadata["supercat"])] * 7).flatten()
labels = LabelEncoder().fit_transform(labels)

fig, axes = plt.subplots(1, 4, figsize=(12, 4))
axes[0].scatter(
    ica_aligned[:, 0],
    ica_aligned[:, 1],
    c=labels,
    cmap="tab10",
)
axes[0].set_xlim(-20, 20)
axes[0].set_ylim(-20, 20)
axes[0].set_title("ICA aligned")
axes[1].scatter(
    ica_unaligned[:, 0], ica_unaligned[:, 1], c=labels, cmap="tab10"
)
axes[1].set_title("ICA unaligned")
axes[1].set_xlim(-20, 20)
axes[1].set_ylim(-20, 20)
axes[2].scatter(pca_aligned[:, 0], pca_aligned[:, 1], c=labels, cmap="tab10")
axes[2].set_title("PCA aligned")
axes[2].set_xlim(-1000, 1000)
axes[2].set_ylim(-1000, 1000)
axes[3].scatter(
    pca_unaligned[:, 0], pca_unaligned[:, 1], c=labels, cmap="tab10"
)
axes[3].set_title("PCA unaligned")
axes[3].set_xlim(-1000, 1000)
axes[3].set_ylim(-1000, 1000)
# Set box aspect ratio
for ax in axes:
    ax.set_aspect("equal")
    ax.set_box_aspect(1)
plt.tight_layout()
plt.show()
