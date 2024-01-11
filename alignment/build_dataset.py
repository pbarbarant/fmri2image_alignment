# %%
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

metadata = pd.read_csv(
    "/data/parietal/store3/work/pbarbara/fmri2image_alignment/data/NSD/nsd_stim_info_merged.csv"
)

# Build the unaligned dataset
# Start with empty columns
df = pd.DataFrame(
    columns=["subject", "image_path", "fmri_path", "split", "shared1000"]
)
subjects = list(range(1, 9))

for sub in subjects:
    print(f"Processing subject {sub}")
    # Create the subject folder
    sub_fmri_folder = Path(
        f"/data/parietal/store3/work/pbarbara/fmri2image_alignment/data/NSD/dataset_unaligned/fMRI/sub-0{sub}"
    )
    sub_fmri_folder.mkdir(parents=True, exist_ok=True)

    # Get the fmri data
    fmri_data = np.load(
        f"/data/parietal/store3/work/pbarbara/fmri2image_alignment/data/NSD/masked_subjects/sub-0{sub}.npy"
    )
    N = fmri_data.shape[0]

    # Get the image data
    image_folder = Path(
        "/data/parietal/store3/work/pbarbara/fmri2image_alignment/data/NSD/stimuli/"
    )
    assert image_folder.exists(), "Image folder does not exist"

    # Select the subjects rep columns
    metadata_subject = metadata[
        [
            f"subject{sub}_rep0",
            f"subject{sub}_rep1",
            f"subject{sub}_rep2",
        ]
    ]

    for idx in tqdm(range(N)):
        fmri = fmri_data[idx]
        # Find the row and column that contains idx+1
        row_id = metadata_subject.isin([idx + 1]).any(axis=1)
        # Get the nsdId
        nsdId = metadata[row_id].nsdId.values[0]
        shared1000 = metadata[row_id].shared1000.values[0]
        image_path = image_folder / f"{nsdId}.jpg"
        # Save the fmri
        fmri_path = sub_fmri_folder / f"{idx}.npy"
        # np.save(fmri_path, fmri)
        # Concatenate the row
        row = pd.DataFrame(
            {
                "subject": sub,
                "image_path": image_path,
                "fmri_path": fmri_path,
                "split": "train" if sub >= 2 else "test",
                "shared1000": shared1000,
            },
            index=[0],
        )
        df = pd.concat([df, row], ignore_index=True)

# Save the dataframe
df.to_csv(
    "/data/parietal/store3/work/pbarbara/fmri2image_alignment/data/NSD/dataset_unaligned/dataset.csv",
    index=False,
)


# Build the aligned dataset
# Start with empty columns
df = pd.DataFrame(
    columns=["subject", "image_path", "fmri_path", "split", "shared1000"]
)
subjects = list(range(1, 9))
target = "sub-01"

for sub in subjects:
    print(f"Processing subject {sub}")
    # Create the subject folder
    sub_fmri_folder = Path(
        f"/data/parietal/store3/work/pbarbara/fmri2image_alignment/data/NSD/dataset_aligned/fMRI/sub-0{sub}"
    )
    sub_fmri_folder.mkdir(parents=True, exist_ok=True)

    if f"sub-0{sub}" == target:
        # Get the fmri data
        fmri_data = np.load(
            f"/data/parietal/store3/work/pbarbara/fmri2image_alignment/data/NSD/masked_subjects/sub-0{sub}.npy"
        )
    else:
        # Get the fmri data
        fmri_data = np.load(
            f"/data/parietal/store3/work/pbarbara/fmri2image_alignment/data/NSD/projected_features/sub-0{sub}_{target}.npy"
        )
    N = fmri_data.shape[0]

    # Get the image data
    image_folder = Path(
        "/data/parietal/store3/work/pbarbara/fmri2image_alignment/data/NSD/stimuli/"
    )
    assert image_folder.exists(), "Image folder does not exist"

    # Select the subjects rep columns
    metadata_subject = metadata[
        [
            f"subject{sub}_rep0",
            f"subject{sub}_rep1",
            f"subject{sub}_rep2",
        ]
    ]

    for idx in tqdm(range(N)):
        fmri = fmri_data[idx]
        # Find the row and column that contains idx+1
        row_id = metadata_subject.isin([idx + 1]).any(axis=1)
        # Get the nsdId
        nsdId = metadata[row_id].nsdId.values[0]
        shared1000 = metadata[row_id].shared1000.values[0]
        image_path = image_folder / f"{nsdId}.jpg"
        # Save the fmri
        fmri_path = sub_fmri_folder / f"{idx}.npy"
        # np.save(fmri_path, fmri)
        # Concatenate the row
        row = pd.DataFrame(
            {
                "subject": sub,
                "image_path": image_path,
                "fmri_path": fmri_path,
                "split": "train" if sub >= 2 else "test",
                "shared1000": shared1000,
            },
            index=[0],
        )
        df = pd.concat([df, row], ignore_index=True)

# Save the dataframe
df.to_csv(
    "/data/parietal/store3/work/pbarbara/fmri2image_alignment/data/NSD/dataset_aligned/dataset.csv",
    index=False,
)
