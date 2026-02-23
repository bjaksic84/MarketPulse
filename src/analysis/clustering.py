"""
Stock behavior clustering analysis.

Segments stocks into behavioral groups using unsupervised learning,
satisfying the college clustering requirement in a financially meaningful way.

Methods:
- K-Means with elbow method + silhouette score
- DBSCAN for density-based comparison
- PCA for 2D visualization
"""

import logging
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class StockClusterAnalyzer:
    """Cluster stocks by behavioral characteristics.

    Computes summary features for each stock (returns, volatility,
    Sharpe ratio, max drawdown, beta, etc.) then clusters them
    into interpretable groups.

    Parameters
    ----------
    market_config : optional
        Market config for data fetching.
    benchmark_ticker : str
        Benchmark ticker for beta calculation (default: SPY).
    """

    # Summary features computed for each stock
    FEATURE_DESCRIPTIONS = {
        "mean_return": "Annualized mean daily return",
        "volatility": "Annualized standard deviation of daily returns",
        "sharpe_ratio": "Annualized Sharpe ratio (return / volatility)",
        "max_drawdown": "Maximum drawdown over the full period",
        "avg_volume_usd": "Average daily dollar volume (log scale)",
        "beta": "Beta vs benchmark (market sensitivity)",
        "skewness": "Skewness of daily returns",
        "kurtosis": "Excess kurtosis of daily returns",
        "up_day_ratio": "Fraction of positive return days",
        "avg_daily_range": "Average (high-low)/close as %",
    }

    def __init__(
        self,
        benchmark_ticker: str = "SPY",
    ):
        self.benchmark_ticker = benchmark_ticker
        self.features_df: Optional[pd.DataFrame] = None
        self.scaled_features: Optional[np.ndarray] = None
        self.scaler: Optional[StandardScaler] = None
        self.pca: Optional[PCA] = None
        self.pca_coords: Optional[pd.DataFrame] = None
        self.kmeans_labels: Optional[np.ndarray] = None
        self.dbscan_labels: Optional[np.ndarray] = None

    def compute_stock_features(
        self,
        data: Dict[str, pd.DataFrame],
        benchmark_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Compute summary behavioral features for each stock.

        Parameters
        ----------
        data : Dict[str, pd.DataFrame]
            Preprocessed OHLCV DataFrames keyed by ticker.
            Each must have columns: close, returns, high, low, volume.
        benchmark_data : pd.DataFrame, optional
            Benchmark OHLCV for beta calculation.

        Returns
        -------
        pd.DataFrame
            Shape (n_tickers, n_features). Index = ticker names.
        """
        records = []

        for ticker, df in data.items():
            if len(df) < 100:
                logger.warning(f"Skipping {ticker}: insufficient data ({len(df)} rows)")
                continue

            try:
                features = self._compute_single_stock_features(
                    ticker, df, benchmark_data
                )
                records.append(features)
            except Exception as e:
                logger.warning(f"Feature computation failed for {ticker}: {e}")

        self.features_df = pd.DataFrame(records).set_index("ticker")
        logger.info(
            f"Computed features for {len(self.features_df)} stocks, "
            f"{len(self.features_df.columns)} features each"
        )

        return self.features_df

    def _compute_single_stock_features(
        self,
        ticker: str,
        df: pd.DataFrame,
        benchmark: Optional[pd.DataFrame],
    ) -> Dict:
        """Compute behavioral features for a single stock."""
        returns = df["returns"].dropna()
        n_days = len(returns)

        # Annualized return & volatility
        mean_ret = returns.mean() * 252
        vol = returns.std() * np.sqrt(252)

        # Sharpe ratio
        sharpe = mean_ret / vol if vol > 0 else 0

        # Maximum drawdown
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = cum_returns / rolling_max - 1
        max_dd = drawdowns.min()

        # Average dollar volume (log scale)
        if "volume" in df.columns and "close" in df.columns:
            dollar_vol = (df["volume"] * df["close"]).mean()
            avg_vol_usd = np.log10(max(dollar_vol, 1))
        else:
            avg_vol_usd = 0

        # Beta vs benchmark
        beta = 0
        if benchmark is not None and "returns" in benchmark.columns:
            # Align dates
            common_idx = returns.index.intersection(benchmark["returns"].index)
            if len(common_idx) > 50:
                bench_ret = benchmark["returns"].loc[common_idx]
                stock_ret = returns.loc[common_idx]
                cov = np.cov(stock_ret, bench_ret)[0, 1]
                bench_var = bench_ret.var()
                beta = cov / bench_var if bench_var > 0 else 0

        # Higher moments
        skew = returns.skew()
        kurt = returns.kurtosis()

        # % of up days
        up_ratio = (returns > 0).mean()

        # Average daily range as % of close
        if "high" in df.columns and "low" in df.columns:
            daily_range = ((df["high"] - df["low"]) / df["close"]).mean()
        else:
            daily_range = 0

        return {
            "ticker": ticker,
            "mean_return": mean_ret,
            "volatility": vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "avg_volume_usd": avg_vol_usd,
            "beta": beta,
            "skewness": skew,
            "kurtosis": kurt,
            "up_day_ratio": up_ratio,
            "avg_daily_range": daily_range,
        }

    def run_kmeans(
        self,
        k_range: range = range(2, 8),
        optimal_k: Optional[int] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """Run K-Means clustering with elbow method.

        Parameters
        ----------
        k_range : range
            Range of k values to try for elbow method.
        optimal_k : int, optional
            Force a specific k instead of auto-selecting.

        Returns
        -------
        Tuple[np.ndarray, Dict]
            (labels, metrics) where metrics contains inertias and silhouette scores.
        """
        if self.features_df is None:
            raise RuntimeError("Run compute_stock_features() first.")

        # Scale features
        self.scaler = StandardScaler()
        self.scaled_features = self.scaler.fit_transform(self.features_df)

        # Elbow method + silhouette scores
        inertias = []
        silhouette_scores = []

        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(self.scaled_features)
            inertias.append(km.inertia_)
            sil = silhouette_score(self.scaled_features, labels)
            silhouette_scores.append(sil)
            logger.info(f"  K={k}: inertia={km.inertia_:.1f}, silhouette={sil:.3f}")

        # Auto-select optimal k (max silhouette)
        if optimal_k is None:
            best_idx = np.argmax(silhouette_scores)
            optimal_k = list(k_range)[best_idx]
            logger.info(
                f"Auto-selected K={optimal_k} "
                f"(silhouette={silhouette_scores[best_idx]:.3f})"
            )

        # Final fit with optimal k
        final_km = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        self.kmeans_labels = final_km.fit_predict(self.scaled_features)
        self.features_df["kmeans_cluster"] = self.kmeans_labels

        metrics = {
            "k_range": list(k_range),
            "inertias": inertias,
            "silhouette_scores": silhouette_scores,
            "optimal_k": optimal_k,
            "final_silhouette": silhouette_score(
                self.scaled_features, self.kmeans_labels
            ),
        }

        return self.kmeans_labels, metrics

    def run_dbscan(
        self,
        eps: float = 1.5,
        min_samples: int = 2,
    ) -> np.ndarray:
        """Run DBSCAN clustering for comparison.

        Parameters
        ----------
        eps : float
            Maximum distance between samples in a cluster.
        min_samples : int
            Minimum samples to form a dense region.

        Returns
        -------
        np.ndarray
            Cluster labels (-1 = noise).
        """
        if self.scaled_features is None:
            self.scaler = StandardScaler()
            self.scaled_features = self.scaler.fit_transform(self.features_df)

        db = DBSCAN(eps=eps, min_samples=min_samples)
        self.dbscan_labels = db.fit_predict(self.scaled_features)
        self.features_df["dbscan_cluster"] = self.dbscan_labels

        n_clusters = len(set(self.dbscan_labels)) - (1 if -1 in self.dbscan_labels else 0)
        n_noise = (self.dbscan_labels == -1).sum()
        logger.info(f"DBSCAN: {n_clusters} clusters, {n_noise} noise points")

        return self.dbscan_labels

    def compute_pca(self, n_components: int = 2) -> pd.DataFrame:
        """Compute PCA for visualization.

        Returns
        -------
        pd.DataFrame
            Columns: [PC1, PC2, ticker, kmeans_cluster, dbscan_cluster]
        """
        if self.scaled_features is None:
            raise RuntimeError("Run clustering first.")

        self.pca = PCA(n_components=n_components)
        coords = self.pca.fit_transform(self.scaled_features)

        self.pca_coords = pd.DataFrame(
            coords,
            columns=[f"PC{i + 1}" for i in range(n_components)],
            index=self.features_df.index,
        )

        if "kmeans_cluster" in self.features_df.columns:
            self.pca_coords["kmeans_cluster"] = self.features_df["kmeans_cluster"]
        if "dbscan_cluster" in self.features_df.columns:
            self.pca_coords["dbscan_cluster"] = self.features_df["dbscan_cluster"]

        explained = self.pca.explained_variance_ratio_
        logger.info(
            f"PCA: {explained[0]:.1%} + {explained[1]:.1%} = "
            f"{sum(explained):.1%} variance explained"
        )

        return self.pca_coords

    def interpret_clusters(self) -> pd.DataFrame:
        """Generate interpretable cluster profiles.

        Returns per-cluster mean of each feature, with human-readable labels.
        """
        if self.features_df is None or "kmeans_cluster" not in self.features_df.columns:
            raise RuntimeError("Run run_kmeans() first.")

        feature_cols = [
            c for c in self.features_df.columns
            if c not in ["kmeans_cluster", "dbscan_cluster"]
        ]

        profiles = (
            self.features_df.groupby("kmeans_cluster")[feature_cols]
            .mean()
            .round(4)
        )

        # Add cluster size
        profiles["n_stocks"] = self.features_df.groupby("kmeans_cluster").size()

        # Add suggested label based on characteristics
        labels = []
        for _, row in profiles.iterrows():
            label = self._suggest_cluster_label(row)
            labels.append(label)
        profiles["suggested_label"] = labels

        return profiles

    def _suggest_cluster_label(self, row: pd.Series) -> str:
        """Suggest a human-readable label for a cluster based on its characteristics."""
        labels = []

        if row.get("volatility", 0) > 0.35:
            labels.append("High-Volatility")
        elif row.get("volatility", 0) < 0.15:
            labels.append("Low-Volatility")

        if row.get("mean_return", 0) > 0.15:
            labels.append("Growth")
        elif row.get("mean_return", 0) < -0.05:
            labels.append("Declining")

        if row.get("beta", 0) > 1.3:
            labels.append("Aggressive")
        elif row.get("beta", 0) < 0.7:
            labels.append("Defensive")

        if row.get("sharpe_ratio", 0) > 1.0:
            labels.append("Quality")

        return " / ".join(labels) if labels else "Moderate"

    # ──────────────────── Visualization ────────────────────

    def plot_elbow(self, metrics: Dict, save_path: Optional[str] = None) -> plt.Figure:
        """Plot elbow method and silhouette scores."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        k_range = metrics["k_range"]

        # Inertia (elbow)
        ax1.plot(k_range, metrics["inertias"], "b-o")
        ax1.set_xlabel("Number of Clusters (K)")
        ax1.set_ylabel("Inertia (Within-Cluster Sum of Squares)")
        ax1.set_title("Elbow Method")
        ax1.axvline(
            x=metrics["optimal_k"], color="red", linestyle="--",
            label=f"Optimal K={metrics['optimal_k']}"
        )
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Silhouette
        ax2.plot(k_range, metrics["silhouette_scores"], "g-o")
        ax2.set_xlabel("Number of Clusters (K)")
        ax2.set_ylabel("Silhouette Score")
        ax2.set_title("Silhouette Analysis")
        ax2.axvline(
            x=metrics["optimal_k"], color="red", linestyle="--",
            label=f"Optimal K={metrics['optimal_k']}"
        )
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.suptitle("K-Means Cluster Selection", fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def plot_clusters_interactive(
        self,
        cluster_col: str = "kmeans_cluster",
    ) -> go.Figure:
        """Create an interactive Plotly scatter plot of PCA clusters."""
        if self.pca_coords is None:
            self.compute_pca()

        df = self.pca_coords.copy()
        df["ticker"] = df.index
        df["cluster"] = df[cluster_col].astype(str)

        # Add feature info for hover
        if self.features_df is not None:
            for col in ["mean_return", "volatility", "sharpe_ratio", "beta"]:
                if col in self.features_df.columns:
                    df[col] = self.features_df[col].round(3)

        fig = px.scatter(
            df,
            x="PC1",
            y="PC2",
            color="cluster",
            hover_name="ticker",
            hover_data=["mean_return", "volatility", "sharpe_ratio", "beta"],
            title="Stock Clusters (PCA Projection)",
            labels={
                "PC1": f"PC1 ({self.pca.explained_variance_ratio_[0]:.1%} var)",
                "PC2": f"PC2 ({self.pca.explained_variance_ratio_[1]:.1%} var)",
                "cluster": "Cluster",
            },
            template="plotly_white",
        )

        fig.update_traces(marker=dict(size=12, line=dict(width=1, color="white")))
        fig.update_layout(width=900, height=600)

        return fig

    def plot_cluster_profiles(
        self,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot radar/parallel coordinates of cluster profiles."""
        profiles = self.interpret_clusters()

        feature_cols = [
            c for c in profiles.columns
            if c not in ["n_stocks", "suggested_label"]
        ]

        # Normalize for radar chart (0-1 scale)
        normalized = profiles[feature_cols].copy()
        for col in feature_cols:
            col_min = normalized[col].min()
            col_max = normalized[col].max()
            denom = col_max - col_min
            if denom > 0:
                normalized[col] = (normalized[col] - col_min) / denom
            else:
                normalized[col] = 0.5

        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(feature_cols))
        width = 0.8 / len(profiles)

        for i, (cluster_id, row) in enumerate(normalized.iterrows()):
            offset = (i - len(profiles) / 2 + 0.5) * width
            label = profiles.loc[cluster_id, "suggested_label"]
            n = profiles.loc[cluster_id, "n_stocks"]
            ax.bar(
                x + offset,
                row[feature_cols].values,
                width=width,
                label=f"Cluster {cluster_id}: {label} (n={n})",
                alpha=0.8,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(feature_cols, rotation=45, ha="right")
        ax.set_ylabel("Normalized Value")
        ax.set_title("Cluster Profiles")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig
