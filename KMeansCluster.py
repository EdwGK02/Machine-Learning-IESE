"""
Part 2.3 - Clustering Application (scikit-learn).

Loads the 1,200-record motorcycle dataset (6 numerical variables), applies
preprocessing (StandardScaler) and K-Means (scikit-learn) to discover
groups of motorcycles with similar technical characteristics, then exposes
everything the Flask templates need: the labeled table, the per-cluster
summary (size + centroid, in original units), the silhouette score, a 2D
scatter plot, and short interpretation text for each cluster.
"""

import os
import io
import base64

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "motorcycles_clustering.csv")
df = pd.read_csv(DATA_PATH)

FEATURES = [
    "engine_displacement_cc",
    "engine_power_hp",
    "weight_kg",
    "price_cop",
    "fuel_consumption_kmpl",
    "max_speed_kmph",
]

FEATURE_LABELS = {
    "engine_displacement_cc": "Displacement (cc)",
    "engine_power_hp": "Power (HP)",
    "weight_kg": "Weight (kg)",
    "price_cop": "Price (COP)",
    "fuel_consumption_kmpl": "Fuel Economy (km/L)",
    "max_speed_kmph": "Max Speed (km/h)",
}

K = 4  # matches the 4 segments the group's brainstorm identified
RANDOM_STATE = 42

N_RECORDS = len(df)

# --- Preprocessing: standardize so displacement/price (large scales) don't
#     dominate the Euclidean distance used internally by K-Means. ---
X = df[FEATURES].to_numpy(dtype=float)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- K-Means (scikit-learn) ---
kmeans = KMeans(n_clusters=K, init="k-means++", n_init=10, max_iter=300, random_state=RANDOM_STATE)
labels = kmeans.fit_predict(X_scaled)

df["cluster"] = labels + 1  # 1-indexed for display

SILHOUETTE_SCORE = round(float(silhouette_score(X_scaled, labels)), 4)

# --- Cluster centroids, converted back to original units for interpretability ---
centroids_scaled = kmeans.cluster_centers_
centroids_original = scaler.inverse_transform(centroids_scaled)

# --- Order clusters by displacement so labels read from "smallest" to "biggest" ---
order = np.argsort(centroids_original[:, FEATURES.index("engine_displacement_cc")])
rank_of = {old: new for new, old in enumerate(order)}
df["cluster"] = df["cluster"].apply(lambda c: rank_of[c - 1] + 1)
centroids_original = centroids_original[order]

CLUSTER_PROFILES = [
    {
        "name": "Urban / Commuter",
        "description": "Small-displacement, lightweight, fuel-efficient and affordable motorcycles built for daily city commuting.",
        "business_note": "The lowest price and highest fuel economy of all segments make this the entry-level, high-volume market for daily city transport.",
    },
    {
        "name": "Naked / Mixed-Use",
        "description": "Mid-range displacement and price, balancing power and efficiency for both city and short-trip riding.",
        "business_note": "Centroid values sit between Urban and Sport on every dimension, consistent with a do-it-all bike for riders who want more power without sacrificing affordability.",
    },
    {
        "name": "Sport / High-Performance",
        "description": "High power-to-weight ratio and top speed, aimed at performance riders; higher price, lower fuel economy.",
        "business_note": "Highest top speed and power-to-weight ratio in the dataset, paired with the steepest fuel-economy penalty — buyers here pay for performance, not efficiency.",
    },
    {
        "name": "Touring / Adventure",
        "description": "Large displacement and the highest weight, built for long-distance comfort and stability rather than outright speed.",
        "business_note": "Despite comparable displacement to Sport, this segment is far heavier and slightly slower, reflecting a design priority of long-distance comfort over acceleration.",
    },
][:K]

CLUSTER_COLORS = ["#4ecdc4", "#ff6b9d", "#ffb347", "#a78bfa", "#6bcB77"][:K]

CLUSTER_SUMMARY = []
for i in range(K):
    mask = df["cluster"] == (i + 1)
    centroid = centroids_original[i]
    CLUSTER_SUMMARY.append({
        "cluster": i + 1,
        "name": CLUSTER_PROFILES[i]["name"] if i < len(CLUSTER_PROFILES) else f"Cluster {i+1}",
        "description": CLUSTER_PROFILES[i]["description"] if i < len(CLUSTER_PROFILES) else "",
        "business_note": CLUSTER_PROFILES[i]["business_note"] if i < len(CLUSTER_PROFILES) else "",  # <-- nueva línea
        "count": int(mask.sum()),
        "pct": round(100 * mask.sum() / N_RECORDS, 1),
        "centroid": {
            # ... (esto no cambia)
        },
    })


def get_sample_table(n=25):
    """A representative sample of labeled records for the 'table of records and clusters' requirement."""
    sample = df.sample(n=n, random_state=RANDOM_STATE).sort_values("cluster")
    return sample[["motorcycle_id"] + FEATURES + ["cluster"]].to_dict(orient="records")


def build_cluster_plot():
    """
    2D scatter: displacement (cc) vs price (COP) -- the two variables with the
    clearest separation and the same axes used in the manual exercise, which
    makes the manual-vs-application comparison easy to follow in the report.
    Centroids are shown converted back to these two original-unit axes.
    """
    fig, ax = plt.subplots(figsize=(8, 5.5))

    x_feat, y_feat = "engine_displacement_cc", "price_cop"
    for i in range(K):
        mask = df["cluster"] == (i + 1)
        ax.scatter(df.loc[mask, x_feat], df.loc[mask, y_feat], alpha=0.45, s=14,
                   color=CLUSTER_COLORS[i], label=f"Cluster {i+1}: {CLUSTER_SUMMARY[i]['name']}")

    cx = [c["centroid"]["engine_displacement_cc"] for c in CLUSTER_SUMMARY]
    cy = [c["centroid"]["price_cop"] for c in CLUSTER_SUMMARY]
    ax.scatter(cx, cy, color="black", marker="X", s=240, edgecolor="white",
               linewidth=1.5, label="Centroids", zorder=5)

    ax.set_title(f"K-Means Clustering of {N_RECORDS} Motorcycles (k={K})")
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


def build_silhouette_note():
    if SILHOUETTE_SCORE >= 0.5:
        return "Strong, well-separated clustering structure."
    elif SILHOUETTE_SCORE >= 0.25:
        return "Reasonable clustering structure with some overlap between neighboring segments."
    else:
        return "Weak structure; clusters overlap considerably."


CLUSTER_PLOT_B64 = build_cluster_plot()
SAMPLE_TABLE = get_sample_table()
SILHOUETTE_NOTE = build_silhouette_note()
