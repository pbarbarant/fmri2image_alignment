# Optimal Transport Strategies for Overcoming Inter-Subject Variability in fMRI Reconstruction

This repository hosts the code developed for the project titled "Optimal Transport Strategies for Overcoming Inter-Subject Variability in fMRI Reconstruction" carried out as part of the Advanced Machine Learning class in 2024 at ENSAE Paris.

---

## Abstract

Recent advancements in generative AI have significantly improved image reconstruction based on functional Magnetic Resonance Imaging (fMRI). However, the persistent challenge of inter-subject variability in large fMRI datasets remains. This study investigates the potential use of optimal transport as a functional alignment tool to address this issue. We explore the novel research question of whether this approach can enhance the performance of generative models in fMRI stimuli reconstruction. Our findings not only reveal the ability of optimal transport in mitigating inter-subject variability but also shows its potential to transfer image semantics between different subjects. This work paves the way for advancing neuroimaging research, offering insights into the intricacies of neural activity representations and paving the way for more precise and personalized fMRI reconstructions across diverse subject populations.

---

## Instructions for running the experiments

1. Download the [NSD](https://naturalscenesdataset.org) dataset and place the file ``nsd_stim_info_merged.csv`` in the ``data/NSD`` folder. Download the [COCO](https://cocodataset.org/) dataset in the ``data/COCO``folder
2. Launch the data preprocessing steps:

   ```
   cd data/NSD
   python coco_image_fetcher.py
   python mask_dataset.py
   python preprocess_alignment_data.py
   cd ..

   ```
3. Compute the alignment mappings:

   ```
   cd alignment
   python compute_mappings.py
   python project_source_to_target.py
   cd ..
   ```
4. Compute the low-dimensional embeddings:

   ```
   cd feature_extraction
   python extract_features.py
   python build_dataset.py
   cd ..
   ```

   (Optional) Plot the UMAP components:

   ```
   cd plotting_utils.py
   python umap_plot.py
   cd ..
   ```
5. Train the diffusion model:

   ```
   cd diffusion
   python trainer_ddpm.py
   ```
6. Train the fMRI encoders:

   ```
   python trainer_aligned_model.py
   python trainer_unaligned_model.py
   ```
7. Evaluate the models visually:

   ```
   python aligned_visual_eval.py
   python unaligned_visual_eval.py
   ```
8. Compute the evaluation metrics for both models:

   ```
   python aligned_metric_eval.py
   python unaligned_metric_eval.py
   ```

---
