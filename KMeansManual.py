import os
import io
import base64

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "motorcycles_manual.csv")
df = pd.read_csv(DATA_PATH)

X_raw = df[["engine_displacement_cc", "price_cop"]].to_numpy(dtype=float)


MEAN = X_raw.mean(axis=0)
STD = X_raw.std(axis=0)
X = (X_raw - MEAN) / STD

CLUSTER_NAMES = ["Cluster 1 (Urban)", "Cluster 2 (Mid-size)", "Cluster 3 (High-performance)"]
CLUSTER_COLORS = ["#4ecdc4", "#ff6b9d", "#ffb347"]


INITIAL_IDX_RAW = [
    int(df["engine_displacement_cc"].sub(125).abs().idxmin()),   # near 125cc -> urban
    int(df["engine_displacement_cc"].sub(400).abs().idxmin()),   # near 400cc -> mid-size
    int(df["engine_displacement_cc"].sub(1000).abs().idxmin()),  # near 1000cc -> high-performance
]
INITIAL_CENTROIDS_RAW = X_raw[INITIAL_IDX_RAW].copy()


def _to_std(centroids_raw):
    return (centroids_raw - MEAN) / STD


def _to_raw(centroids_std):
    return centroids_std * STD + MEAN


def _euclidean(points, centroids):
    diff = points[:, None, :] - centroids[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=2))


def _variance(points, centroids, labels):
    per_cluster = []
    for k in range(len(centroids)):
        mask = labels == k
        if mask.sum() == 0:
            per_cluster.append(0.0)
            continue
        d2 = ((points[mask] - centroids[k]) ** 2).sum(axis=1)
        per_cluster.append(float(d2.mean()))
    overall = float(np.mean([( (points[i]-centroids[labels[i]])**2 ).sum() for i in range(len(points))]))
    return per_cluster, overall


def run_manual_kmeans(n_iterations=3):
    centroids_std = _to_std(INITIAL_CENTROIDS_RAW.copy())
    iterations = []
    plots = {"initial": _plot_state(X_raw, None, INITIAL_CENTROIDS_RAW, "Initial Records and Centroids")}

    for it in range(1, n_iterations + 1):
        distances = _euclidean(X, centroids_std) 
        labels = distances.argmin(axis=1)

        rows = []
        for i in range(len(df)):
            rows.append({
                "id": df["motorcycle_id"].iloc[i],
                "cc": int(df["engine_displacement_cc"].iloc[i]),
                "price": int(df["price_cop"].iloc[i]),
                "d1": round(float(distances[i, 0]), 3),
                "d2": round(float(distances[i, 1]), 3),
                "d3": round(float(distances[i, 2]), 3),
                "cluster": int(labels[i]) + 1,
            })

        new_centroids_std = np.array([
            X[labels == k].mean(axis=0) if (labels == k).sum() > 0 else centroids_std[k]
            for k in range(len(centroids_std))
        ])

        per_cluster_var, overall_var = _variance(X, centroids_std, labels)
        counts = [int((labels == k).sum()) for k in range(len(centroids_std))]

        new_centroids_raw = _to_raw(new_centroids_std)
        plot_b64 = _plot_state(X_raw, labels, new_centroids_raw, f"Iteration {it}: Clusters and Updated Centroids")

        iterations.append({
            "n": it,
            "rows": rows,
            "counts": counts,
            "centroids_raw": [
                {"cc": round(float(c[0]), 1), "price": int(round(c[1]))} for c in new_centroids_raw
            ],
            "variance_per_cluster": [round(v, 4) for v in per_cluster_var],
            "variance_overall": round(overall_var, 4),
            "plot": plot_b64,
        })

        centroids_std = new_centroids_std

    return {
        "initial_centroids": [
            {"cc": int(c[0]), "price": int(c[1])} for c in INITIAL_CENTROIDS_RAW
        ],
        "initial_plot": plots["initial"],
        "iterations": iterations,
        "cluster_names": CLUSTER_NAMES,
        "n_records": len(df),
    }


def _plot_state(points_raw, labels, centroids_raw, title):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    if labels is None:
        ax.scatter(points_raw[:, 0], points_raw[:, 1], alpha=0.6, color="#b8b8b8", s=28, label="Motorcycles")
    else:
        for k in range(len(centroids_raw)):
            mask = labels == k
            ax.scatter(points_raw[mask, 0], points_raw[mask, 1], alpha=0.65,
                       color=CLUSTER_COLORS[k], s=28, label=CLUSTER_NAMES[k])

    ax.scatter(centroids_raw[:, 0], centroids_raw[:, 1], color="black", marker="X",
               s=220, edgecolor="white", linewidth=1.5, label="Centroids", zorder=5)

    ax.set_title(title)
    ax.set_xlabel("Engine Displacement (cc)")
    ax.set_ylabel("Price (COP)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

MANUAL_RESULT = run_manual_kmeans()
