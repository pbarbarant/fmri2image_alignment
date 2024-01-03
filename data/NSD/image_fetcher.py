# %%
from pathlib import Path
import os
import shutil
from tqdm import tqdm
from PIL import Image
import pandas as pd


def retrieve_metadata(metadata_path, metadata_output):
    # Check if the file exists
    if not metadata_path.exists():
        raise FileNotFoundError("NSD metadata file not found.")

    # Use shutil to copy the file
    shutil.copy(src=metadata_path, dst=metadata_output)

    # Open the metadata file
    metadata = pd.read_csv(metadata_output)

    return metadata


def fetch_and_process_images(metadata, output_dir):
    for i in tqdm(range(len(metadata))):
        idx = metadata["nsdId"].iloc[i]
        cocoId = metadata.loc[idx, "cocoId"]
        cocoSplit = metadata.loc[idx, "cocoSplit"]
        cropBox = eval(metadata.loc[idx, "cropBox"])

        # Open the image
        coco_path = f"/data/parietal/store3/work/pbarbara/fmri2image_alignment/data/COCO/{cocoSplit}/{cocoId:012d}.jpg"
        img = Image.open(coco_path)

        # Get the crop box
        width, height = img.size
        crop_left = width * cropBox[2]
        crop_right = width * (1 - cropBox[3])
        crop_top = height * cropBox[0]
        crop_bottom = height * (1 - cropBox[1])

        # Crop the image based on the crop box
        cropped_image = img.crop(
            (crop_left, crop_top, crop_right, crop_bottom)
        )

        # Save the image
        cropped_image.save(output_dir / f"{idx}.jpg")


if __name__ == "__main__":
    metadata_path = Path(
        "/data/parietal/store3/data/natural_scenes/info/nsd_stim_info_merged.csv"
    )
    metadata_output = Path(
        "/data/parietal/store3/work/pbarbara/fmri2image_alignment/data/NSD/nsd_stim_info_merged.csv"
    )
    metadata = retrieve_metadata(metadata_path)

    output_dir = Path(
        "/data/parietal/store3/work/pbarbara/fmri2image_alignment/data/NSD/unaligned/images"
    )
    if not output_dir.exists():
        os.makedirs(output_dir)
    fetch_and_process_images(metadata, output_dir)
