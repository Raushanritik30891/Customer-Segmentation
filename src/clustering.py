"""
Clustering Module
------------------
Implements KMeans clustering workflow:
  1. Elbow method for optimal k
  2. Silhouette score evaluation
  3. Final model training
  4. Cluster labeling with business-friendly names
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


def elbow_analysis(X_scaled: np.ndarray, k_range: range = range(2, 11)) -> dict:
    """
    Run KMeans for k in k_range, record inertia and silhouette score.
    Returns dict with lists for plotting.
    """
    print("[INFO] Running elbow + silhouette analysis...")
    inertias = []
    silhouettes = []
    ks = list(k_range)

    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sil = silhouette_score(X_scaled, labels, sample_size=min(5000, X_scaled.shape[0]))
        silhouettes.append(sil)
        print(f"  k={k}  inertia={km.inertia_:,.0f}  silhouette={sil:.4f}")

    return {"k": ks, "inertia": inertias, "silhouette": silhouettes}


def find_optimal_k(elbow_results: dict) -> int:
    """
    Pick optimal k using the elbow heuristic (max second-derivative of inertia)
    combined with peak silhouette, then reconcile.
    """
    ks = elbow_results["k"]
    inertias = elbow_results["inertia"]
    silhouettes = elbow_results["silhouette"]

    # Second derivative of inertia
    if len(inertias) >= 3:
        diffs = np.diff(inertias)
        diffs2 = np.diff(diffs)
        elbow_idx = np.argmax(diffs2) + 1  # +1 because diff reduces length
        elbow_k = ks[elbow_idx]
    else:
        elbow_k = ks[0]

    # Max silhouette
    sil_k = ks[np.argmax(silhouettes)]

    # Prefer silhouette; clamp to range [4, 6] for business usability
    optimal = sil_k
    if not (4 <= optimal <= 6):
        optimal = elbow_k
    optimal = max(4, min(6, optimal))

    print(f"[INFO] Elbow suggests k={elbow_k}, silhouette peak at k={sil_k}  → using k={optimal}")
    return optimal


def train_kmeans(X_scaled: np.ndarray, k: int) -> KMeans:
    """Train final KMeans model."""
    print(f"[INFO] Training KMeans with k={k}...")
    model = KMeans(n_clusters=k, random_state=42, n_init=20, max_iter=500)
    model.fit(X_scaled)
    print(f"[INFO] Final inertia: {model.inertia_:,.2f}")
    return model


def assign_segment_names(cluster_stats: pd.DataFrame) -> dict:
    """
    Data-driven assignment of business-friendly segment names.
    Uses ranked quintiles of Monetary, Recency, Frequency.

    Rules (applied in priority order):
      1. Lowest Recency + Highest Monetary → Premium
      2. Lowest Recency + High Frequency   → Loyal
      3. Highest Monetary (not Premium)    → High Value
      4. Highest Recency (churned)         → At Risk
      5. Low Frequency + Moderate Monetary → Potential
      6. Remaining                         → Regular
    """
    stats = cluster_stats.copy()
    n = len(stats)
    names = {}

    # Rank each cluster (lower rank = better for Monetary/Frequency; better = lower Recency)
    stats["rec_rank"] = stats["Recency_mean"].rank()        # lower = more recent = better
    stats["mon_rank"] = stats["Monetary_mean"].rank(ascending=False)  # higher spend = rank 1
    stats["freq_rank"] = stats["Frequency_mean"].rank(ascending=False)

    assigned = set()
    idx = stats["Cluster"].tolist()

    def best(col, top_n=1, exclude=None):
        excl = exclude or set()
        sub = stats[~stats["Cluster"].isin(excl)]
        return sub.nsmallest(top_n, col)["Cluster"].tolist()

    # 1. Premium: best recency AND best monetary
    premium_cands = set(best("rec_rank")) & set(best("mon_rank"))
    if not premium_cands:
        # Compromise: lowest recency + highest monetary combined score
        stats["pm_score"] = stats["rec_rank"] + stats["mon_rank"]
        premium_cands = {stats.nsmallest(1, "pm_score")["Cluster"].iloc[0]}
    premium = premium_cands.pop()
    names[premium] = "💎 Premium Customers"
    assigned.add(premium)

    # 2. High Value: highest monetary (not premium)
    hv = best("mon_rank", exclude=assigned)[0]
    names[hv] = "🏆 High Value Customers"
    assigned.add(hv)

    # 3. At Risk: highest recency (longest since last purchase)
    at_risk = stats[~stats["Cluster"].isin(assigned)].nlargest(1, "Recency_mean")["Cluster"].iloc[0]
    names[at_risk] = "⚠️ At Risk Customers"
    assigned.add(at_risk)

    # 4. Loyal: best frequency among remaining
    remaining = stats[~stats["Cluster"].isin(assigned)]
    if len(remaining) >= 2:
        loyal = remaining.nsmallest(1, "freq_rank")["Cluster"].iloc[0]
        names[loyal] = "🔄 Loyal Customers"
        assigned.add(loyal)

    # 5. Remaining clusters
    labels_left = ["🌱 Potential Customers", "📦 Regular Customers"]
    i = 0
    for _, row in stats[~stats["Cluster"].isin(assigned)].iterrows():
        names[row["Cluster"]] = labels_left[i % len(labels_left)]
        i += 1

    return names


def build_cluster_profiles(features: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """
    Compute per-cluster statistics and assign segment names.
    Returns enriched dataframe.
    """
    features = features.copy()
    features["Cluster"] = labels

    agg_cols = [
        "Recency", "Frequency", "Monetary",
        "AvgOrderValue", "UniqueProducts", "ActiveDays",
        "CustomerValueScore", "LoyaltyScore"
    ]
    agg_cols = [c for c in agg_cols if c in features.columns]

    # Per-cluster mean
    stats = features.groupby("Cluster")[agg_cols].mean().reset_index()
    stats.columns = ["Cluster"] + [f"{c}_mean" for c in agg_cols]
    stats["CustomerCount"] = features.groupby("Cluster").size().values

    # Assign names
    name_map = assign_segment_names(stats)
    stats["SegmentName"] = stats["Cluster"].map(name_map)
    features["SegmentName"] = features["Cluster"].map(name_map)

    return features, stats


def plot_elbow(elbow_results: dict, optimal_k: int, save_path: str = None):
    """Save elbow + silhouette plot."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(elbow_results["k"], elbow_results["inertia"], "bo-")
    axes[0].axvline(x=optimal_k, color="red", linestyle="--", label=f"Optimal k={optimal_k}")
    axes[0].set_title("Elbow Method")
    axes[0].set_xlabel("Number of Clusters (k)")
    axes[0].set_ylabel("Inertia")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(elbow_results["k"], elbow_results["silhouette"], "go-")
    axes[1].axvline(x=optimal_k, color="red", linestyle="--", label=f"Optimal k={optimal_k}")
    axes[1].set_title("Silhouette Score")
    axes[1].set_xlabel("Number of Clusters (k)")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Saved elbow plot → {save_path}")
    plt.close()


if __name__ == "__main__":
    import sys, joblib
    sys.path.insert(0, ".")
    from src.data_preprocessing import load_data, clean_data
    from src.feature_engineering import build_customer_features, get_feature_matrix

    df = clean_data(load_data("data/Online_Retail.xlsx"))
    features = build_customer_features(df)
    X_scaled, scaler, feat_cols, features = get_feature_matrix(features)

    elbow_results = elbow_analysis(X_scaled)
    k = find_optimal_k(elbow_results)
    model = train_kmeans(X_scaled, k)
    labels = model.labels_

    features, cluster_stats = build_cluster_profiles(features, labels)
    print("\n[CLUSTER PROFILES]")
    print(cluster_stats[["Cluster", "SegmentName", "CustomerCount", "Recency_mean", "Frequency_mean", "Monetary_mean"]])
