"""
MarketPulse Streamlit Dashboard.

Interactive web application for exploring predictions,
model performance, and stock clustering analysis.

Run with: streamlit run app/streamlit_app.py
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.fetcher import YFinanceFetcher
from src.data.market_config import (
    MarketConfig,
    list_available_markets,
    load_market_config,
    load_strategy_config,
)
from src.data.preprocessing import preprocess_ohlcv
from src.features.labels import generate_labels, get_clean_features_and_labels
from src.features.returns import compute_return_features
from src.features.technical import compute_technical_indicators
from src.models.evaluator import MarketPulseEvaluator
from src.models.xgboost_classifier import MarketPulseXGBClassifier
from src.utils.validation import WalkForwardValidator


# ──────────────────── Page Config ────────────────────

st.set_page_config(
    page_title="MarketPulse",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ──────────────────── Sidebar ────────────────────

def render_sidebar():
    """Render sidebar controls and return selected parameters."""
    st.sidebar.title("📈 MarketPulse")
    st.sidebar.markdown("*AI-Driven Price Movement Predictor*")
    st.sidebar.divider()

    # Market selection
    available_markets = list_available_markets()
    market_name = st.sidebar.selectbox(
        "Market",
        available_markets,
        index=0,
        help="Select the financial market to analyze",
    )
    market_config = load_market_config(market_name)

    # Ticker selection
    tickers = market_config.default_tickers
    selected_ticker = st.sidebar.selectbox(
        "Ticker",
        tickers,
        index=0,
        help="Select a specific instrument",
    )

    # Date range
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input(
            "Start",
            datetime.now() - timedelta(days=5 * 365),
        )
    with col2:
        end_date = st.date_input("End", datetime.now())

    # Prediction settings
    st.sidebar.divider()
    st.sidebar.subheader("Prediction Settings")

    horizon = st.sidebar.selectbox(
        "Prediction Horizon (days)",
        [1, 3, 5],
        index=0,
    )

    num_classes = st.sidebar.radio(
        "Classification Mode",
        [2, 3],
        index=1,
        format_func=lambda x: "Binary (UP/DOWN)" if x == 2 else "Ternary (UP/FLAT/DOWN)",
    )

    threshold = st.sidebar.slider(
        "FLAT Zone Threshold (%)",
        min_value=0.5,
        max_value=3.0,
        value=1.0,
        step=0.25,
        disabled=(num_classes == 2),
    )

    # View selection
    st.sidebar.divider()
    view = st.sidebar.radio(
        "View",
        ["Ticker Explorer", "Prediction", "Model Performance", "Clustering"],
    )

    return {
        "market_name": market_name,
        "market_config": market_config,
        "ticker": selected_ticker,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "horizon": horizon,
        "num_classes": num_classes,
        "threshold": threshold / 100,
        "view": view,
    }


# ──────────────────── Data Loading ────────────────────

@st.cache_data(ttl=3600, show_spinner="Fetching market data...")
def load_ticker_data(fmt_ticker: str, start: str, end: str, market_name: str):
    """Fetch and preprocess data for a ticker (cached for 1 hour)."""
    market_config = load_market_config(market_name)
    fetcher = YFinanceFetcher(market_config=market_config)
    raw = fetcher.fetch(fmt_ticker, start=start, end=end)
    if raw.empty:
        return pd.DataFrame()
    df = preprocess_ohlcv(raw, market_config=market_config)
    df = compute_technical_indicators(df)
    df = compute_return_features(df)
    return df


# ──────────────────── Views ────────────────────

def render_ticker_explorer(params):
    """Candlestick chart with technical indicators overlay."""
    st.header(f"📊 {params['ticker']} — Ticker Explorer")

    fmt_ticker = params["market_config"].format_ticker(params["ticker"])

    df = load_ticker_data(
        fmt_ticker, params["start_date"], params["end_date"], params["market_name"]
    )

    if df.empty:
        st.error(f"No data available for {fmt_ticker}")
        return

    # Summary metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Latest Close", f"${df['close'].iloc[-1]:.2f}")
    with col2:
        last_ret = df["returns"].iloc[-1] * 100
        st.metric("Last Return", f"{last_ret:+.2f}%")
    with col3:
        vol = df["returns"].rolling(20).std().iloc[-1] * np.sqrt(252) * 100
        st.metric("20d Volatility", f"{vol:.1f}%")
    with col4:
        rsi = df["rsi_14"].iloc[-1] if "rsi_14" in df.columns else 0
        st.metric("RSI (14)", f"{rsi:.1f}")
    with col5:
        st.metric("Data Points", f"{len(df):,}")

    # Candlestick chart
    fig = go.Figure()

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        name="OHLC",
    ))

    # Bollinger Bands
    show_bb = st.checkbox("Show Bollinger Bands", value=True)
    if show_bb and "bb_upper" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["bb_upper"], mode="lines",
            name="BB Upper", line=dict(color="rgba(173,216,230,0.5)", width=1),
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=df["bb_lower"], mode="lines",
            name="BB Lower", line=dict(color="rgba(173,216,230,0.5)", width=1),
            fill="tonexty", fillcolor="rgba(173,216,230,0.1)",
        ))

    # Moving Averages
    show_sma = st.checkbox("Show SMA (20/50)", value=True)
    if show_sma:
        if "sma_20" in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df["sma_20"], mode="lines",
                name="SMA 20", line=dict(color="orange", width=1),
            ))
        if "sma_50" in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df["sma_50"], mode="lines",
                name="SMA 50", line=dict(color="purple", width=1),
            ))

    fig.update_layout(
        title=f"{fmt_ticker} Price Chart",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        template="plotly_white",
        height=500,
        xaxis_rangeslider_visible=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    # RSI subplot
    if "rsi_14" in df.columns:
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(
            x=df.index, y=df["rsi_14"], mode="lines",
            name="RSI (14)", line=dict(color="blue"),
        ))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
        fig_rsi.update_layout(
            title="RSI (14)",
            height=250,
            template="plotly_white",
            yaxis=dict(range=[0, 100]),
        )
        st.plotly_chart(fig_rsi, use_container_width=True)

    # Volume
    fig_vol = go.Figure()
    colors = ["green" if r >= 0 else "red" for r in df["returns"]]
    fig_vol.add_trace(go.Bar(
        x=df.index, y=df["volume"], marker_color=colors, name="Volume",
    ))
    fig_vol.update_layout(
        title="Volume", height=200, template="plotly_white",
    )
    st.plotly_chart(fig_vol, use_container_width=True)


def render_prediction(params):
    """Prediction panel with SHAP explanation."""
    st.header(f"🔮 {params['ticker']} — Price Prediction")

    fmt_ticker = params["market_config"].format_ticker(params["ticker"])

    df = load_ticker_data(
        fmt_ticker, params["start_date"], params["end_date"], params["market_name"]
    )

    if df.empty:
        st.error(f"No data available for {fmt_ticker}")
        return

    # Generate labels
    labeled_df = generate_labels(
        df,
        horizon=params["horizon"],
        label_type="classification",
        num_classes=params["num_classes"],
        threshold=params["threshold"],
    )

    # Get clean features and labels
    X, y = get_clean_features_and_labels(labeled_df)

    if len(X) < 600:
        st.warning(f"Insufficient data ({len(X)} samples). Need at least 600.")
        return

    # Train a model on all but the last 21 days
    train_end = len(X) - 21
    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_test, y_test = X.iloc[train_end:], y.iloc[train_end:]

    with st.spinner("Training model..."):
        strategy_config = load_strategy_config("short_term")
        model = MarketPulseXGBClassifier.from_strategy_config(strategy_config)
        model.num_classes = params["num_classes"]
        model.fit(X_train, y_train)

    # Latest prediction
    X_latest = X.iloc[[-1]]
    pred = model.predict(X_latest)[0]
    proba = model.predict_proba(X_latest)[0]

    class_names = {0: "DOWN", 1: "FLAT", 2: "UP"} if params["num_classes"] == 3 else {0: "DOWN", 1: "UP"}
    pred_name = class_names[int(pred)]
    confidence = proba.max() * 100

    # Display prediction
    col1, col2, col3 = st.columns(3)

    color_map = {"UP": "🟢", "DOWN": "🔴", "FLAT": "🟡"}
    with col1:
        st.markdown(f"### {color_map.get(pred_name, '⚪')} {pred_name}")
        st.caption(f"Horizon: {params['horizon']} day(s)")
    with col2:
        st.metric("Confidence", f"{confidence:.1f}%")
    with col3:
        st.metric("As of", X_latest.index[-1].strftime("%Y-%m-%d"))

    # Probability breakdown
    st.subheader("Class Probabilities")
    prob_df = pd.DataFrame({
        "Class": [class_names[i] for i in range(len(proba))],
        "Probability": proba,
    })
    st.bar_chart(prob_df.set_index("Class"))

    # Feature importance
    st.subheader("Top Features Driving This Prediction")
    importance = model.get_feature_importance().head(15)
    fig_imp = go.Figure(go.Bar(
        x=importance.values,
        y=importance.index,
        orientation="h",
        marker_color="steelblue",
    ))
    fig_imp.update_layout(
        title="Feature Importance (Gain)",
        xaxis_title="Importance",
        height=400,
        template="plotly_white",
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    # Recent test performance
    st.subheader("Recent Test Period Performance (Last 21 Days)")
    y_pred_test = model.predict(X_test)
    from sklearn.metrics import accuracy_score, classification_report

    acc = accuracy_score(y_test.astype(int), y_pred_test)
    st.metric("Test Accuracy", f"{acc:.1%}")

    report = classification_report(
        y_test.astype(int),
        y_pred_test,
        target_names=[class_names[i] for i in sorted(class_names.keys())],
        output_dict=True,
    )
    st.dataframe(pd.DataFrame(report).T.round(3))


def render_model_performance(params):
    """Walk-forward validation results."""
    st.header(f"📉 {params['ticker']} — Model Performance")

    fmt_ticker = params["market_config"].format_ticker(params["ticker"])

    df = load_ticker_data(
        fmt_ticker, params["start_date"], params["end_date"], params["market_name"]
    )

    if df.empty:
        st.error(f"No data available for {fmt_ticker}")
        return

    # Generate labels
    labeled_df = generate_labels(
        df,
        horizon=params["horizon"],
        label_type="classification",
        num_classes=params["num_classes"],
        threshold=params["threshold"],
    )

    X, y = get_clean_features_and_labels(labeled_df)

    if len(X) < 600:
        st.warning(f"Insufficient data ({len(X)} samples).")
        return

    # Walk-forward validation
    validator = WalkForwardValidator(
        initial_train_days=504,
        test_days=21,
        step_days=21,
    )
    folds = validator.split(X)

    evaluator = MarketPulseEvaluator(num_classes=params["num_classes"])
    strategy_config = load_strategy_config("short_term")

    fold_accuracies = []
    fold_f1s = []
    fold_dates = []
    all_true = []
    all_pred = []

    progress = st.progress(0, text="Running walk-forward validation...")

    for i, fold in enumerate(folds):
        X_train, y_train, X_test, y_test = validator.get_fold_data(X, y, fold)

        model = MarketPulseXGBClassifier.from_strategy_config(strategy_config)
        model.num_classes = params["num_classes"]
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        result = evaluator.evaluate_fold(
            y_true=y_test.values.astype(int),
            y_pred=y_pred,
            y_proba=y_proba,
            fold_number=fold.fold_number,
            train_size=fold.train_size,
        )

        fold_accuracies.append(result.accuracy)
        fold_f1s.append(result.f1)
        fold_dates.append(fold.test_start_date)
        all_true.extend(y_test.values.astype(int))
        all_pred.extend(y_pred)

        progress.progress((i + 1) / len(folds), text=f"Fold {i + 1}/{len(folds)}")

    progress.empty()

    # Summary metrics
    from sklearn.metrics import accuracy_score

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean Accuracy", f"{np.mean(fold_accuracies):.1%}")
    with col2:
        st.metric("Mean F1", f"{np.mean(fold_f1s):.3f}")
    with col3:
        majority_baseline = pd.Series(all_true).value_counts(normalize=True).max()
        st.metric("Baseline (Majority)", f"{majority_baseline:.1%}")
    with col4:
        lift = np.mean(fold_accuracies) - majority_baseline
        st.metric("Lift", f"{lift:+.1%}")

    # Accuracy over time
    st.subheader("Accuracy Over Time (Walk-Forward)")
    perf_df = pd.DataFrame({
        "Date": fold_dates,
        "Accuracy": fold_accuracies,
        "F1": fold_f1s,
    }).set_index("Date")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=perf_df.index, y=perf_df["Accuracy"],
        mode="lines+markers", name="Accuracy", line=dict(color="blue"),
    ))
    fig.add_hline(y=np.mean(fold_accuracies), line_dash="dash", line_color="green",
                  annotation_text=f"Mean: {np.mean(fold_accuracies):.3f}")
    fig.add_hline(y=majority_baseline, line_dash="dot", line_color="red",
                  annotation_text=f"Baseline: {majority_baseline:.3f}")
    fig.update_layout(
        title="Walk-Forward Fold Accuracy",
        yaxis_title="Accuracy",
        template="plotly_white",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Confusion matrix
    st.subheader("Aggregate Confusion Matrix")
    from sklearn.metrics import confusion_matrix
    import plotly.figure_factory as ff

    class_names = (
        ["DOWN", "FLAT", "UP"] if params["num_classes"] == 3 else ["DOWN", "UP"]
    )
    cm = confusion_matrix(all_true, all_pred)
    fig_cm = go.Figure(data=go.Heatmap(
        z=cm,
        x=class_names,
        y=class_names,
        text=cm,
        texttemplate="%{text}",
        colorscale="Blues",
    ))
    fig_cm.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=400,
        template="plotly_white",
    )
    st.plotly_chart(fig_cm, use_container_width=True)


def render_clustering(params):
    """Stock clustering analysis view."""
    st.header("🔬 Stock Behavior Clustering")

    from src.analysis.clustering import StockClusterAnalyzer
    from src.data.preprocessing import preprocess_multiple

    market_config = params["market_config"]
    tickers = market_config.default_tickers
    formatted = market_config.format_tickers(tickers)

    # Fetch data for all tickers
    with st.spinner(f"Fetching data for {len(formatted)} tickers..."):
        fetcher = YFinanceFetcher(market_config=market_config)
        raw_data = fetcher.fetch_multiple(
            formatted, start=params["start_date"], end=params["end_date"]
        )

    if not raw_data:
        st.error("Failed to fetch data")
        return

    # Preprocess
    processed = preprocess_multiple(raw_data, market_config=market_config)

    if len(processed) < 4:
        st.warning(f"Only {len(processed)} tickers available. Need at least 4.")
        return

    # Fetch benchmark
    benchmark_ticker = market_config.format_ticker(market_config.benchmark)
    benchmark_data = processed.get(benchmark_ticker)

    # Cluster
    analyzer = StockClusterAnalyzer(benchmark_ticker=benchmark_ticker)
    features = analyzer.compute_stock_features(processed, benchmark_data)

    st.subheader("Stock Feature Summary")
    st.dataframe(features.round(3), use_container_width=True)

    # K-Means
    n_clusters = st.slider("Number of Clusters (K)", 2, 6, 3)
    labels, metrics = analyzer.run_kmeans(optimal_k=n_clusters)

    # Elbow plot
    col1, col2 = st.columns(2)
    with col1:
        fig_elbow = analyzer.plot_elbow(metrics)
        st.pyplot(fig_elbow)
        plt.close()

    with col2:
        # Cluster profiles
        profiles = analyzer.interpret_clusters()
        st.subheader("Cluster Profiles")
        st.dataframe(profiles.round(3), use_container_width=True)

    # PCA scatter plot
    st.subheader("Cluster Visualization (PCA)")
    analyzer.compute_pca()
    fig_scatter = analyzer.plot_clusters_interactive()
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Cluster membership
    st.subheader("Cluster Membership")
    for cluster_id in sorted(features["kmeans_cluster"].unique()):
        members = features[features["kmeans_cluster"] == cluster_id].index.tolist()
        label = profiles.loc[cluster_id, "suggested_label"]
        st.markdown(f"**Cluster {cluster_id}** — *{label}*: {', '.join(members)}")


# ──────────────────── Main ────────────────────

def main():
    params = render_sidebar()

    view = params["view"]

    if view == "Ticker Explorer":
        render_ticker_explorer(params)
    elif view == "Prediction":
        render_prediction(params)
    elif view == "Model Performance":
        render_model_performance(params)
    elif view == "Clustering":
        render_clustering(params)


if __name__ == "__main__":
    main()
