# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Paths to pca and ica features
ica_aligned_path = Path(
    "/data/parietal/store3/work/pbarbara/fmri2image_alignment/data/NSD/ica_sub-01/transformed_features.npy"
)
ica_unaligned_path = Path(
    "/data/parietal/store3/work/pbarbara/fmri2image_alignment/data/NSD/ica_sub-01/transformed_features.npy"
)
pca_aligned_path = Path(
    "/data/parietal/store3/work/pbarbara/fmri2image_alignment/data/NSD/pca_sub-01/transformed_features.npy"
)
pca_unaligned_path = Path(
    "/data/parietal/store3/work/pbarbara/fmri2image_alignment/data/NSD/pca_sub-01/transformed_features.npy"
)

ica_aligned = np.load(ica_aligned_path)
ica_unaligned = np.load(ica_unaligned_path)
pca_aligned = np.load(pca_aligned_path)
pca_unaligned = np.load(pca_unaligned_path)


labels = np.concatenate([np.ones(902) * i for i in range(1, 9)])

fig, axes = plt.subplots(1, 4, figsize=(12, 4))
axes[0].scatter(ica_aligned[:, 0], ica_aligned[:, 1], c=labels)
axes[0].set_title("ICA aligned")
axes[1].scatter(ica_unaligned[:, 0], ica_unaligned[:, 1], c=labels)
axes[1].set_title("ICA unaligned")
axes[2].scatter(pca_aligned[:, 0], pca_aligned[:, 1], c=labels)
axes[2].set_title("PCA aligned")
axes[3].scatter(pca_unaligned[:, 0], pca_unaligned[:, 1], c=labels)
axes[3].set_title("PCA unaligned")
plt.tight_layout()
plt.legend()
plt.show()
