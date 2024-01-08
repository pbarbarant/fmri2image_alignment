# %%
import numpy as np
from fugw.utils import load_mapping
from pathlib import Path


if __name__ == "__main__":
    target = "sub-01"
    sources = [f"sub-{i:02d}" for i in range(2, 9)]
    mappings_folder = Path(
        "/data/parietal/store3/work/pbarbara/fmri2image_alignment/alignment/mappings/"
    )
    output_folder = Path(
        "/data/parietal/store3/work/pbarbara/fmri2image_alignment/data/NSD/projected_features/"
    )
    if not output_folder.exists():
        output_folder.mkdir(parents=True)
    for source in sources:
        print(f"Projecting {source} to {target}")
        mapping = load_mapping(mappings_folder / f"{source}_{target}.pkl")
        print("Mapping loaded")
        source_features = np.load(
            f"/data/parietal/store3/work/pbarbara/fmri2image_alignment/data/NSD/masked_subjects/{source}.npy"
        )
        print("Features loaded")
        projected_features = mapping.transform(source_features)
        print("Features projected")
        np.save(output_folder / f"{source}_{target}.npy", projected_features)
