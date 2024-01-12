# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from utils import load_features_from_model, load_labels_share1000


def evaluate_model(model, n_components=1024, verbose=True):
    """Evaluate the given model using a linear SVM classifier.

    Parameters
    ----------
    model : str
        Model to evaluate. Must be one of "pca_unaligned", "pca_aligned",
        "ica_unaligned", "ica_aligned".
    n_components : int, optional
        Number of components to use. Defaults to 1024.
    verbose : bool, optional
        Whether to print progress information, by default True
    """
    X_train, X_test = load_features_from_model(model=model)
    labels = load_labels_share1000()
    y_train = np.concatenate([labels] * 7).flatten()
    y_test = labels.flatten()

    # Scale data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Fit model
    if verbose:
        print("Fitting model...")
    clf = LinearSVC()
    clf.fit(X_train, y_train)

    if verbose:
        print("Evaluating model...")

    # Predict
    y_pred = clf.predict(X_test)

    # Evaluate
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=clf.classes_
    )
    disp.plot(
        cmap="Blues",
        xticks_rotation="vertical",
    )
    plt.title(f"Confusion matrix for {model}")

    # Save plot
    plt.savefig(
        f"/data/parietal/store3/work/pbarbara/fmri2image_alignment/figures/{model}.png"
    )

    # Save report to csv
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_dir = Path(
        "/data/parietal/store3/work/pbarbara/fmri2image_alignment/classification/reports"
    )
    report_dir.mkdir(exist_ok=True)
    report_df.to_csv(report_dir / f"{model}.csv")


if __name__ == "__main__":
    for model in [
        "pca_unaligned",
        "pca_aligned",
        "ica_unaligned",
        "ica_aligned",
    ]:
        evaluate_model(model=model, verbose=False)
