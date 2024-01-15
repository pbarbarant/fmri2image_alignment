# %%
from pathlib import Path

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from nilearn import datasets, image, plotting, surface


def plot_surface_map(
    volume, vmin=-5, vmax=5, colorbar=True, cmap="coolwarm", **kwargs
):
    # Load fsaverage surface
    fsaverage = datasets.fetch_surf_fsaverage("fsaverage5")
    # Project to surface
    surface_map = surface.vol_to_surf(volume, fsaverage.pial_left)

    # Plot
    plotting.plot_surf(
        fsaverage.pial_left,
        surface_map,
        hemi="right",
        view="medial",
        colorbar=colorbar,
        threshold=0.5,
        cmap=cmap,
        symmetric_cbar=True,
        vmax=vmax,
        vmin=vmin,
        bg_map=fsaverage.sulc_left,
        bg_on_data=True,
        darkness=0.5,
        **kwargs,
    )


def generate_fig(contrasts):
    fig = plt.figure(figsize=(3 * len(contrasts), 3))
    fig.suptitle(
        "Inter-individual differences in brain activity (NSD dataset)"
    )
    grid_spec = gridspec.GridSpec(1, len(contrasts), figure=fig)

    for i, contrast in enumerate(contrasts):
        ax = fig.add_subplot(grid_spec[0, i], projection="3d")
        ax.set_title(f"Subject {i+1}")
        plot_surface_map(contrast, axes=ax, colorbar=False, vmax=10, vmin=-10)

    # Add colorbar
    ax = fig.add_subplot(grid_spec[0, :])
    ax.axis("off")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="1%")
    fig.add_axes(cax)
    fig.colorbar(
        mpl.cm.ScalarMappable(
            norm=mpl.colors.Normalize(vmin=-10, vmax=10), cmap="coolwarm"
        ),
        cax=cax,
    )
    return fig


if __name__ == "__main__":
    # Select the first 4 subjects
    subjects = [f"sub-{i:02d}" for i in range(1, 5)]
    subject_paths = [
        Path(
            f"/data/parietal/store3/data/natural_scenes/curated_3mm/{sub}.nii.gz"
        )
        for sub in subjects
    ]

    # Check that the paths are correct
    for path in subject_paths:
        assert path.exists(), f"Path {path} does not exist"

    # Load contrast 2616 for each subject
    contrasts = [image.index_img(path, index=2616) for path in subject_paths]
    fig = generate_fig(contrasts)
    fig.tight_layout()
    fig.savefig(
        "/data/parietal/store3/work/pbarbara/fmri2image_alignment/figures/inter_individual_diff.png",
        dpi=300,
        bbox_inches="tight",
    )
