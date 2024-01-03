import pandas as pd
import shutil


def retrieve_metadata(source, target):
    # Check if the file exists
    if not source.exists():
        raise FileNotFoundError("NSD metadata file not found.")

    # Use shutil to copy the file
    shutil.copy(src=source, dst=target)

    # Open the metadata file
    metadata = pd.read_csv(target)

    return metadata


def load_metadata(metadata_path):
    # Open the metadata file
    metadata = pd.read_csv(metadata_path)

    return metadata
