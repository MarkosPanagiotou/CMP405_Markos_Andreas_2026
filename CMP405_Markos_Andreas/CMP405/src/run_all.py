"""CMP405 - MNIST: supervised, unsupervised and semi-supervised learning.

This script runs every required experiment and writes tables/figures under results/.
The MNIST labels are used according to the assignment rules:
- A1: all training labels are used (supervised).
- B: K-Means fit receives no labels; labels are inspected only after fit for evaluation.
- C: exactly 5% of training labels are exposed via stratified sampling.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    pairwise_distances,
    precision_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import LabelPropagation
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIG = RESULTS / "figures"
TAB = RESULTS / "tables"
for p in (FIG, TAB):
    p.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


def summary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def save_report(y_true: np.ndarray, y_pred: np.ndarray, filename: str) -> None:
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    pd.DataFrame(rep).T.to_csv(TAB / filename, index=True)


def save_confusion(y_true: np.ndarray, y_pred: np.ndarray, title: str, filename: str, figsize=(7, 6)) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=np.arange(10), cmap="Blues", colorbar=False,
        values_format="d", ax=ax
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(FIG / filename, dpi=180)
    plt.close(fig)


def main() -> None:
    timings: dict[str, float] = {}
    metrics: dict[str, object] = {}

    d = np.load(DATA / "mnist_standard.npz")
    X_train_img, y_train = d["X_train"], d["y_train"]
    X_test_img, y_test = d["X_test"], d["y_test"]
    assert X_train_img.shape == (60000, 28, 28)
    assert X_test_img.shape == (10000, 28, 28)

    X_train = X_train_img.reshape(-1, 784).astype(np.float32) / 255.0
    X_test = X_test_img.reshape(-1, 784).astype(np.float32) / 255.0

    # Dataset examples
    fig, axes = plt.subplots(2, 5, figsize=(8, 3.6))
    chosen = [np.flatnonzero(y_train == digit)[0] for digit in range(10)]
    for ax, idx in zip(axes.flat, chosen):
        ax.imshow(X_train_img[idx], cmap="gray")
        ax.set_title(f"Label {int(y_train[idx])}")
        ax.axis("off")
    fig.suptitle("MNIST examples - one sample per class")
    fig.tight_layout()
    fig.savefig(FIG / "mnist_examples.png", dpi=180)
    plt.close(fig)

    print("[1/6] Loaded MNIST", flush=True)

    # A1 - Supervised Random Forest
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=200,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    timings["supervised_fit_seconds"] = time.time() - t0
    pred_sup = rf.predict(X_test)
    metrics["supervised_random_forest"] = summary_metrics(y_test, pred_sup)
    save_report(y_test, pred_sup, "supervised_classification_report.csv")
    save_confusion(y_test, pred_sup, "A1 - Random Forest on MNIST test set", "confusion_supervised.png")

    print("[2/6] A1 complete", flush=True)

    # A2 - New digitally drawn digits, same fitted RF (NO retraining)
    ext = np.load(DATA / "external_digits" / "external_digits.npz")
    X_ext_img, y_ext = ext["X"], ext["y"]
    X_ext = X_ext_img.reshape(-1, 784).astype(np.float32) / 255.0
    pred_ext = rf.predict(X_ext)
    metrics["external_digits"] = summary_metrics(y_ext, pred_ext)
    save_report(y_ext, pred_ext, "external_classification_report.csv")
    save_confusion(y_ext, pred_ext, "A2 - Confusion matrix on 20 new digits", "confusion_external.png", figsize=(6.5, 5.5))

    ext_rows = []
    raw_files = sorted((DATA / "external_digits" / "raw").glob("*.png"), key=lambda p: (int(p.stem.split('_')[1]), int(p.stem.split('_')[2])))
    for i, (yt, yp) in enumerate(zip(y_ext, pred_ext)):
        ext_rows.append({"sample": raw_files[i].name, "true_label": int(yt), "prediction": int(yp), "correct": bool(yt == yp)})
    pd.DataFrame(ext_rows).to_csv(TAB / "external_predictions.csv", index=False)

    fig, axes = plt.subplots(4, 5, figsize=(9, 7))
    for ax, img, yt, yp in zip(axes.flat, X_ext_img, y_ext, pred_ext):
        ax.imshow(img, cmap="gray")
        ax.set_title(f"True {int(yt)} | Pred {int(yp)}", fontsize=9)
        ax.axis("off")
    fig.suptitle("A2 - New digitally drawn digits after MNIST-compatible preprocessing")
    fig.tight_layout()
    fig.savefig(FIG / "external_digits_predictions.png", dpi=180)
    plt.close(fig)

    print("[3/6] A2 complete", flush=True)

    # PCA is an unsupervised dimensionality-reduction preprocessing step.
    t0 = time.time()
    pca = PCA(n_components=20, svd_solver="randomized", random_state=RANDOM_STATE)
    with threadpool_limits(limits=8):
        Z_train = pca.fit_transform(X_train)
        Z_test = pca.transform(X_test)
    timings["pca_fit_transform_seconds"] = time.time() - t0
    metrics["pca_explained_variance_ratio_sum"] = float(pca.explained_variance_ratio_.sum())

    # B - K-Means k=10, labels NOT used during fit
    t0 = time.time()
    kmeans = KMeans(
        n_clusters=10,
        init="k-means++",
        n_init=3,
        max_iter=100,
        tol=1e-3,
        random_state=RANDOM_STATE,
        algorithm="lloyd",
    )
    print("[4/6] PCA complete", flush=True)
    with threadpool_limits(limits=8):
        clusters_train = kmeans.fit_predict(Z_train)
        clusters_test = kmeans.predict(Z_test)
    timings["kmeans_fit_seconds"] = time.time() - t0

    distances_to_center = np.linalg.norm(Z_train - kmeans.cluster_centers_[clusters_train], axis=1)
    center_distances = pairwise_distances(kmeans.cluster_centers_)
    nonzero_center_distances = center_distances[np.triu_indices_from(center_distances, k=1)]
    sil = silhouette_score(Z_train, clusters_train, sample_size=3000, random_state=RANDOM_STATE)

    # Labels are used only now for post-hoc interpretation/evaluation.
    cluster_counts = pd.crosstab(pd.Series(clusters_train, name="cluster"), pd.Series(y_train, name="digit"))
    cluster_counts.to_csv(TAB / "cluster_composition_counts.csv")
    cluster_purity_per = cluster_counts.max(axis=1) / cluster_counts.sum(axis=1)
    purity = float(cluster_counts.max(axis=1).sum() / len(y_train))
    posthoc_map = {int(k): int(cluster_counts.loc[k].idxmax()) for k in cluster_counts.index}
    mapped_train = np.array([posthoc_map[int(k)] for k in clusters_train])
    mapped_test = np.array([posthoc_map[int(k)] for k in clusters_test])

    metrics["kmeans"] = {
        "k": 10,
        "inertia_sse": float(kmeans.inertia_),
        "sse_note": "For Euclidean K-Means, sklearn inertia is the within-cluster Sum of Squared Errors (SSE).",
        "cohesion_mean_distance_to_centroid": float(distances_to_center.mean()),
        "cohesion_mean_squared_distance": float(kmeans.inertia_ / len(Z_train)),
        "separation_min_centroid_distance": float(nonzero_center_distances.min()),
        "separation_mean_centroid_distance": float(nonzero_center_distances.mean()),
        "silhouette_coefficient_sample_3000": float(sil),
        "posthoc_cluster_purity": purity,
        "posthoc_mapped_accuracy_test": float(accuracy_score(y_test, mapped_test)),
        "posthoc_cluster_to_digit_map": posthoc_map,
    }

    # Cluster composition heatmap
    comp = cluster_counts.div(cluster_counts.sum(axis=1), axis=0).to_numpy()
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(comp, aspect="auto", cmap="viridis")
    ax.set_xlabel("True digit (used only for post-hoc analysis)")
    ax.set_ylabel("K-Means cluster")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_title("B - Digit composition inside each unlabeled K-Means cluster")
    fig.colorbar(im, ax=ax, label="Proportion within cluster")
    fig.tight_layout()
    fig.savefig(FIG / "cluster_composition.png", dpi=180)
    plt.close(fig)

    # Reconstructed centroids for interpretation
    centers_pixels = np.clip(pca.inverse_transform(kmeans.cluster_centers_), 0, 1).reshape(10, 28, 28)
    fig, axes = plt.subplots(2, 5, figsize=(8, 3.8))
    for k, ax in enumerate(axes.flat):
        ax.imshow(centers_pixels[k], cmap="gray")
        ax.set_title(f"Cluster {k} -> ~{posthoc_map[k]}")
        ax.axis("off")
    fig.suptitle("B - K-Means centroids reconstructed to image space")
    fig.tight_layout()
    fig.savefig(FIG / "kmeans_centroids.png", dpi=180)
    plt.close(fig)

    print("[5/6] K-Means complete", flush=True)

    # C - exactly 5% labels, selected stratified.
    indices = np.arange(len(y_train))
    labeled_idx, unlabeled_idx = train_test_split(
        indices, train_size=0.05, stratify=y_train, random_state=RANDOM_STATE
    )
    assert len(labeled_idx) == 3000
    label_dist = np.bincount(y_train[labeled_idx], minlength=10)
    pd.DataFrame({"digit": np.arange(10), "labeled_count": label_dist}).to_csv(TAB / "labeled_5pct_distribution.csv", index=False)
    metrics["semi_supervised_split"] = {
        "labeled_count": int(len(labeled_idx)),
        "unlabeled_count": int(len(unlabeled_idx)),
        "labeled_fraction": float(len(labeled_idx) / len(y_train)),
        "labeled_count_by_digit": {str(i): int(v) for i, v in enumerate(label_dist)},
    }

    # Optional baseline: same 5% labels but without unlabeled data.
    print("[5a] split complete", flush=True)
    rf_5 = RandomForestClassifier(n_estimators=200, max_features="sqrt", random_state=RANDOM_STATE, n_jobs=-1)
    rf_5.fit(X_train[labeled_idx], y_train[labeled_idx])
    pred_rf5 = rf_5.predict(X_test)
    print("[5b] 5pct baseline complete", flush=True)
    metrics["supervised_5pct_baseline"] = summary_metrics(y_test, pred_rf5)

    # C1 - Label Propagation (only 5% labels exposed; 95% = -1)
    y_semi = np.full(len(y_train), -1, dtype=np.int16)
    y_semi[labeled_idx] = y_train[labeled_idx]
    t0 = time.time()
    label_prop = LabelPropagation(kernel="knn", n_neighbors=7, max_iter=500, tol=1e-3)
    print("[5c] LP fitting", flush=True)
    with threadpool_limits(limits=8):
        label_prop.fit(Z_train, y_semi)
    print("[5d] LP fit complete", flush=True)
    timings["label_propagation_fit_seconds"] = time.time() - t0
    with threadpool_limits(limits=8):
        pred_lp = label_prop.predict(Z_test)
    print("[5e] LP predict complete", flush=True)
    metrics["label_propagation"] = {
        **summary_metrics(y_test, pred_lp),
        "n_iter": int(label_prop.n_iter_),
        "kernel": "knn",
        "n_neighbors": 7,
        "max_iter": 500,
    }
    save_report(y_test, pred_lp, "label_propagation_classification_report.csv")
    save_confusion(y_test, pred_lp, "C1 - Label Propagation on MNIST test set", "confusion_label_propagation.png")

    print("[5f] LP outputs saved", flush=True)

    # C2 - K-Means + cluster labeling, map clusters using ONLY the 5% labeled samples.
    global_majority = int(np.bincount(y_train[labeled_idx]).argmax())
    cluster_map_5 = {}
    for k in range(10):
        in_cluster_labeled = labeled_idx[clusters_train[labeled_idx] == k]
        if len(in_cluster_labeled) == 0:
            cluster_map_5[k] = global_majority
        else:
            vals, counts = np.unique(y_train[in_cluster_labeled], return_counts=True)
            cluster_map_5[k] = int(vals[np.argmax(counts)])
    pred_cluster_label = np.array([cluster_map_5[int(k)] for k in clusters_test])
    metrics["clustering_plus_cluster_labeling"] = {
        **summary_metrics(y_test, pred_cluster_label),
        "cluster_to_digit_map_from_5pct_labels": cluster_map_5,
    }
    save_report(y_test, pred_cluster_label, "cluster_labeling_classification_report.csv")
    save_confusion(y_test, pred_cluster_label, "C2 - K-Means + Cluster Labeling", "confusion_cluster_labeling.png")

    # Comparison table and chart
    comparison = pd.DataFrame([
        {"method": "Supervised Random Forest (100% labels)", **metrics["supervised_random_forest"]},
        {"method": "New handwritten digits (same RF)", **metrics["external_digits"]},
        {"method": "K-Means (post-hoc mapped)", "accuracy": metrics["kmeans"]["posthoc_mapped_accuracy_test"], "precision_macro": np.nan, "recall_macro": np.nan, "f1_macro": np.nan},
        {"method": "Label Propagation (5% labels)", **{k: metrics["label_propagation"][k] for k in ["accuracy","precision_macro","recall_macro","f1_macro"]}},
        {"method": "K-Means + Cluster Labeling (5%)", **metrics["clustering_plus_cluster_labeling"]},
        {"method": "Supervised RF baseline (5% only)", **metrics["supervised_5pct_baseline"]},
    ])
    # Remove dict column leaked from cluster labeling row, if present.
    if "cluster_to_digit_map_from_5pct_labels" in comparison.columns:
        comparison = comparison.drop(columns=["cluster_to_digit_map_from_5pct_labels"])
    comparison.to_csv(TAB / "method_comparison.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 4.7))
    plot_df = comparison.iloc[:5]
    ax.barh(plot_df["method"], plot_df["accuracy"] * 100)
    ax.set_xlabel("Accuracy (%)")
    ax.set_xlim(0, 100)
    ax.set_title("Comparison of required CMP405 methods")
    for i, v in enumerate(plot_df["accuracy"] * 100):
        ax.text(v + 0.8, i, f"{v:.1f}%", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "method_comparison_accuracy.png", dpi=180)
    plt.close(fig)

    print("[6/6] Semi-supervised complete", flush=True)
    metrics["timings_seconds"] = timings
    with open(RESULTS / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
