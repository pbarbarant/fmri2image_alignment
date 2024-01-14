import numpy as np
import torch
from fugw.mappings import FUGW, FUGWSparse
from fugw.scripts import coarse_to_fine, lmds
from nilearn import masking


class FugwAlignment:
    """Wrapper for FUGW alignment"""

    def __init__(
        self,
        masker,
        method="coarse_to_fine",
        n_samples=300,
        alpha_coarse=0.5,
        rho_coarse=1,
        eps_coarse=1e-6,
        alpha_fine=0.5,
        rho_fine=1,
        eps_fine=1e-6,
        radius=5,
        id_reg=False,
    ) -> None:
        """Initialize FUGW alignment

        Parameters
        ----------
        masker : NiftiMasker
            Masker used to extract features
        method : str, optional
            Method used to compute FUGW alignments, by default "coarse_to_fine"
        n_samples : int, optional
            Number of samples from the embedding, by default 300
        alpha_coarse : float, optional, by default 0.5
        rho_coarse : int, optional, by default 1
        eps_coarse : _type_, optional, by default 1e-6
        alpha_fine : float, optional, by default 0.5
        rho_fine : int, optional, by default 1
        eps_fine : _type_, optional, by default 1e-6
        radius : int, optional
            Radius around the sampled points in mm, by default 5
        id_reg : bool, optional
            Interpolate the resulting mapping with the identity matrix,
            by default False
        """
        self.masker = masker
        self.method = method
        self.n_samples = int(n_samples)
        self.alpha_coarse = alpha_coarse
        self.rho_coarse = rho_coarse
        self.eps_coarse = eps_coarse
        self.alpha_fine = alpha_fine
        self.rho_fine = rho_fine
        self.eps_fine = eps_fine
        self.radius = radius
        self.id_reg = id_reg

    def fit(self, X, Y, verbose=True):
        """Fit FUGW alignment

        Parameters
        ----------
        X : ndarray
            Source features
        Y : ndarray
            Target features
        verbose : bool, optional, by default True

        Returns
        -------
        self : FugwAlignment
            Fitted FUGW alignment
        """

        # Get main connected component of segmentation
        segmentation = (
            masking.compute_background_mask(
                self.masker.mask_img_, connected=True
            ).get_fdata()
            > 0
        )

        if verbose:
            print("Connected component extracted")

        # Compute the embedding of the source and target data
        source_geometry_embeddings = lmds.compute_lmds_volume(
            segmentation,
            k=12,
            n_landmarks=1000,
            anisotropy=(2, 2, 2),
            verbose=verbose,
        ).nan_to_num()
        target_geometry_embeddings = source_geometry_embeddings.clone()
        (
            source_embeddings_normalized,
            source_distance_max,
        ) = coarse_to_fine.random_normalizing(source_geometry_embeddings)
        (
            target_embeddings_normalized,
            target_distance_max,
        ) = coarse_to_fine.random_normalizing(target_geometry_embeddings)

        if verbose:
            print("Embeddings computed")

        source_features_normalized = X / np.linalg.norm(X, axis=1).reshape(
            -1, 1
        )
        target_features_normalized = Y / np.linalg.norm(Y, axis=1).reshape(
            -1, 1
        )

        if verbose:
            print("Features computed")

        if self.method == "dense":
            mapping = FUGW(
                alpha=self.alpha_coarse,
                rho=self.rho_coarse,
                eps=self.eps_coarse,
                reg_mode="independent",
                divergence="kl",
            )

            mapping.fit(
                source_features=source_features_normalized,
                target_features=target_features_normalized,
                source_geometry=source_embeddings_normalized
                @ source_embeddings_normalized.T,
                target_geometry=target_embeddings_normalized
                @ target_embeddings_normalized.T,
                verbose=verbose,
            )

            self.mapping = mapping

        elif self.method == "coarse_to_fine":
            # Subsample vertices as uniformly as possible on the surface
            source_sample = coarse_to_fine.sample_volume_uniformly(
                segmentation,
                embeddings=source_geometry_embeddings,
                n_samples=self.n_samples,
            )
            target_sample = coarse_to_fine.sample_volume_uniformly(
                segmentation,
                embeddings=target_geometry_embeddings,
                n_samples=self.n_samples,
            )

            if verbose:
                print("Samples computed")

            coarse_mapping = FUGW(
                alpha=self.alpha_coarse,
                rho=self.rho_coarse,
                eps=self.eps_coarse,
                reg_mode="independent",
                divergence="kl",
            )

            fine_mapping = FUGWSparse(
                alpha=self.alpha_fine,
                rho=self.rho_fine,
                eps=self.eps_fine,
                reg_mode="independent",
                divergence="kl",
            )

            coarse_to_fine.fit(
                # Source and target's features and embeddings
                source_features=source_features_normalized,
                target_features=target_features_normalized,
                source_geometry_embeddings=source_embeddings_normalized,
                target_geometry_embeddings=target_embeddings_normalized,
                # Parametrize step 1 (coarse alignment between source and target)
                source_sample=source_sample,
                target_sample=target_sample,
                coarse_mapping=coarse_mapping,
                coarse_mapping_solver="mm",
                coarse_mapping_solver_params={
                    "nits_bcd": 50,
                    "nits_uot": 100,
                },
                # Parametrize step 2 (selection of pairs of indices present in
                # fine-grained's sparsity mask)
                coarse_pairs_selection_method="topk",
                source_selection_radius=(self.radius / source_distance_max),
                target_selection_radius=(self.radius / target_distance_max),
                # Parametrize step 3 (fine-grained alignment)
                fine_mapping=fine_mapping,
                fine_mapping_solver="mm",
                fine_mapping_solver_params={
                    "nits_bcd": 20,
                    "nits_uot": 100,
                },
                # Misc
                device=torch.device("cuda:0"),
                verbose=verbose,
            )

            self.mapping = fine_mapping

        return self

    def transform(self, X):
        """Project features using the fitted FUGW alignment

        Parameters
        ----------
        X : ndarray
            Source features

        Returns
        -------
        ndarray
            Projected features
        """

        # If id_reg is True, interpolate the resulting
        # mapping with the identity matrix
        if self.id_reg is True:
            transformed_features = (self.mapping.transform(X) + X) / 2
        else:
            transformed_features = self.mapping.transform(X)
        return self.masker.inverse_transform(transformed_features)
