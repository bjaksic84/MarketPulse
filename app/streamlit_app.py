"""
MarketPulse Streamlit Dashboard — Phase 4.

Interactive web application for exploring predictions,
model performance, stock clustering, and regression analysis.

Features:
- Multi-market support with market-specific strategies
- Ensemble models (XGBoost + LightGBM)
- Market-adaptive + macro features
- Regime detection overlay
- Classification AND regression views
- Strategy selector

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
from src.features.market_adaptive import compute_market_adaptive_features
from src.features.macro import compute_macro_features
from src.analysis.regime import detect_regime, REGIME_LABELS
from src.models.evaluator import MarketPulseEvaluator
from src.models.regression_evaluator import RegressionEvaluator
from src.models.xgboost_classifier import MarketPulseXGBClassifier
from src.models.ensemble import MarketPulseEnsemble
from src.utils.validation import WalkForwardValidator


# ──────────────────── Page Config ────────────────────

st.set_page_config(
    page_title="MarketPulse",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ──────────────────── Strategy Mapping ────────────────────

MARKET_STRATEGY_MAP = {
    "indices": "indices_short_term",
    "stocks": "short_term",
}


def _get_available_strategies():
    """List all available strategy config files."""
    config_dir = PROJECT_ROOT / "config" / "strategies"
    strategies = []
    if config_dir.exists():
        for f in sorted(config_dir.glob("*.yaml")):
            strategies.append(f.stem)
    return strategies if strategies else ["short_term"]


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

    # Strategy selection (Phase 4)
    available_strategies = _get_available_strategies()
    default_strategy = MARKET_STRATEGY_MAP.get(market_name, "short_term")
    default_idx = (
        available_strategies.index(default_strategy)
        if default_strategy in available_strategies
        else 0
    )
    strategy_name = st.sidebar.selectbox(
        "Strategy",
        available_strategies,
        index=default_idx,
        help="Strategy profile (auto-selected per market)",
    )
    strategy_config = load_strategy_config(strategy_name)

    # Date range
    years = strategy_config.get("data", {}).get("years_of_history", 5)
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input(
            "Start",
            datetime.now() - timedelta(days=years * 365),
        )
    with col2:
        end_date = st.date_input("End", datetime.now())

    # Prediction settings
    st.sidebar.divider()
    st.sidebar.subheader("Prediction Settings")

    horizon_options = strategy_config.get("horizon_days", [1, 3, 5])
    default_horizon = strategy_config.get("default_horizon", 1)
    horizon = st.sidebar.selectbox(
        "Prediction Horizon (days)",
        horizon_options,
        index=(
            horizon_options.index(default_horizon)
            if default_horizon in horizon_options
            else 0
        ),
    )

    label_type = strategy_config.get("label_type", "classification")

    if label_type == "classification":
        num_classes = st.sidebar.radio(
            "Classification Mode",
            [2, 3],
            index=0,
            format_func=lambda x: (
                "Binary (UP/DOWN)" if x == 2 else "Ternary (UP/FLAT/DOWN)"
            ),
        )

        threshold_default = strategy_config.get("threshold", 0.01)
        is_adaptive = isinstance(threshold_default, str) and threshold_default.lower() == "adaptive"

        if is_adaptive:
            use_adaptive_threshold = st.sidebar.checkbox(
                "Adaptive Threshold (balanced classes)",
                value=True,
                help="Compute per-ticker thresholds from return percentiles for balanced UP/FLAT/DOWN classes",
            )
            if use_adaptive_threshold:
                threshold = "adaptive"
            else:
                # Fallback: manual slider
                threshold = st.sidebar.slider(
                    "FLAT Zone Threshold (%)",
                    min_value=0.5,
                    max_value=5.0,
                    value=1.0,
                    step=0.25,
                    disabled=(num_classes == 2),
                )
                threshold = threshold / 100
        else:
            threshold = st.sidebar.slider(
                "FLAT Zone Threshold (%)",
                min_value=0.5,
                max_value=5.0,
                value=float(threshold_default) * 100,
                step=0.25,
                disabled=(num_classes == 2),
            )
    else:
        num_classes = 2
        threshold = 0.0

    # Model mode
    st.sidebar.divider()
    use_ensemble = st.sidebar.checkbox(
        "Use Ensemble (XGB + LGB)",
        value=strategy_config.get("ensemble", {}).get("enabled", True),
        help="Combine XGBoost + LightGBM for more robust predictions",
    )

    # View selection
    st.sidebar.divider()
    views = ["Ticker Explorer", "Prediction", "Model Performance", "Clustering"]
    if label_type == "regression" or "regression" in strategy_name:
        views.insert(3, "Regression")
    view = st.sidebar.radio("View", views)

    return {
        "market_name": market_name,
        "market_config": market_config,
        "ticker": selected_ticker,
        "strategy_name": strategy_name,
        "strategy_config": strategy_config,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "horizon": horizon,
        "num_classes": num_classes,
        "threshold": threshold if threshold == "adaptive" else threshold / 100,
        "label_type": label_type,
        "use_ensemble": use_ensemble,
        "view": view,
    }


# ──────────────────── Data Loading (Enhanced) ────────────────────

@st.cache_data(ttl=3600, show_spinner="Fetching & enriching data...")
def load_enriched_data(
    fmt_ticker: str,
    start: str,
    end: str,
    market_name: str,
    strategy_name: str,
):
    """Fetch, preprocess, compute ALL features (cached 1 hour).

    Includes: technical + returns + market-adaptive + macro + regime.
    """
    market_config = load_market_config(market_name)
    strategy_config = load_strategy_config(strategy_name)
    fetcher = YFinanceFetcher(market_config=market_config)
    raw = fetcher.fetch(fmt_ticker, start=start, end=end)
    if raw.empty:
        return pd.DataFrame()

    df = preprocess_ohlcv(raw, market_config=market_config)
    df = compute_technical_indicators(df)
    df = compute_return_features(df)

    # Phase 3: market-adaptive features
    df = compute_market_adaptive_features(
        df, market_name=market_name, strategy_config=strategy_config
    )

    # Phase 3: regime detection
    df = detect_regime(df, market_name=market_name)

    # Phase 4: macro features (only if strategy uses them)
    strategy_features = strategy_config.get("features", [])
    if "macro" in strategy_features:
        df = compute_macro_features(df, strategy_config=strategy_config)

    return df


# ──────────────────── Views ────────────────────

def render_ticker_explorer(params):
    """Candlestick chart with technical indicators + regime overlay."""
    st.header(f"📊 {params['ticker']} — Ticker Explorer")

    fmt_ticker = params["market_config"].format_ticker(params["ticker"])

    df = load_enriched_data(
        fmt_ticker,
        params["start_date"],
        params["end_date"],
        params["market_name"],
        params["strategy_name"],
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
        regime = (
            df["regime_label"].iloc[-1]
            if "regime_label" in df.columns
            else "N/A"
        )
        st.metric(
            "Regime",
            regime.title() if isinstance(regime, str) else "N/A",
        )

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

    # Regime overlay (Phase 3)
    if "regime" in df.columns:
        show_regime = st.checkbox("Show Market Regime", value=True)
        if show_regime:
            fig_regime = go.Figure()
            colors_map = {0: "red", 1: "gray", 2: "green"}
            for regime_val, color in colors_map.items():
                mask = df["regime"] == regime_val
                if mask.any():
                    fig_regime.add_trace(go.Scatter(
                        x=df.index[mask],
                        y=df["close"][mask],
                        mode="markers",
                        marker=dict(size=3, color=color),
                        name=REGIME_LABELS.get(regime_val, "?"),
                    ))
            fig_regime.update_layout(
                title="Market Regime (color = regime)",
                height=300,
                template="plotly_white",
            )
            st.plotly_chart(fig_regime, use_container_width=True)


def _build_model(params):
    """Build model based on ensemble setting and strategy config."""
    strategy_config = params["strategy_config"]
    if params["use_ensemble"]:
        return MarketPulseEnsemble.from_strategy_config(strategy_config)
    else:
        return MarketPulseXGBClassifier.from_strategy_config(strategy_config)


def render_prediction(params):
    """Prediction panel with SHAP explanation (ensemble-aware)."""
    st.header(f"🔮 {params['ticker']} — Price Prediction")

    fmt_ticker = params["market_config"].format_ticker(params["ticker"])

    df = load_enriched_data(
        fmt_ticker,
        params["start_date"],
        params["end_date"],
        params["market_name"],
        params["strategy_name"],
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
        st.warning(f"Insufficient data ({len(X)} samples). Need at least 600.")
        return

    # Feature selection (same as model performance view)
    strategy = params["strategy_config"]
    fs_cfg = strategy.get("feature_selection", {})
    fs_method = fs_cfg.get("method", "none")
    if fs_method != "none":
        from src.features.feature_selection import select_features_pipeline
        max_features = fs_cfg.get("max_features", 15)
        corr_threshold = fs_cfg.get("corr_threshold", 0.90)
        val_init = strategy.get("validation", {}).get("initial_train_days", 504)
        n_select = min(val_init, len(X))
        selected_features, _ = select_features_pipeline(
            X.iloc[:n_select], y.iloc[:n_select],
            max_features=max_features,
            corr_threshold=corr_threshold,
            method=fs_method,
        )
        X = X[selected_features]

    # Train on all but last 21 days
    train_end = len(X) - 21
    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_test, y_test = X.iloc[train_end:], y.iloc[train_end:]

    with st.spinner("Training model..."):
        model = _build_model(params)
        model.fit(X_train, y_train)

    # Latest prediction
    X_latest = X.iloc[[-1]]
    pred = model.predict(X_latest)[0]
    proba = model.predict_proba(X_latest)[0]

    class_names = (
        {0: "DOWN", 1: "FLAT", 2: "UP"}
        if params["num_classes"] == 3
        else {0: "DOWN", 1: "UP"}
    )
    pred_name = class_names[int(pred)]
    confidence = proba.max() * 100

    # Display prediction
    col1, col2, col3, col4 = st.columns(4)

    color_map = {"UP": "🟢", "DOWN": "🔴", "FLAT": "🟡"}
    with col1:
        st.markdown(f"### {color_map.get(pred_name, '⚪')} {pred_name}")
        st.caption(f"Horizon: {params['horizon']} day(s)")
    with col2:
        st.metric("Confidence", f"{confidence:.1f}%")
    with col3:
        st.metric("As of", X_latest.index[-1].strftime("%Y-%m-%d"))
    with col4:
        model_type = "Ensemble" if params["use_ensemble"] else "XGBoost"
        st.metric("Model", model_type)

    # Ensemble agreement (Phase 3)
    if params["use_ensemble"] and hasattr(model, "get_agreement_score"):
        agreement = model.get_agreement_score(X_latest)
        if len(agreement) > 0:
            ag_pct = agreement[0] * 100
            st.info(
                f"Ensemble agreement: {ag_pct:.0f}% — "
                + (
                    "All models agree"
                    if ag_pct == 100
                    else "Models partially disagree"
                )
            )

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
    """Walk-forward validation with ensemble + adaptive features."""
    st.header(f"📉 {params['ticker']} — Model Performance")

    fmt_ticker = params["market_config"].format_ticker(params["ticker"])

    df = load_enriched_data(
        fmt_ticker,
        params["start_date"],
        params["end_date"],
        params["market_name"],
        params["strategy_name"],
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

    # Use strategy-specific validation settings
    strategy = params["strategy_config"]

    # Feature selection (reduces overfitting from too many features)
    fs_cfg = strategy.get("feature_selection", {})
    fs_method = fs_cfg.get("method", "none")
    if fs_method != "none":
        from src.features.feature_selection import select_features_pipeline
        max_features = fs_cfg.get("max_features", 15)
        corr_threshold = fs_cfg.get("corr_threshold", 0.90)
        val_init = strategy.get("validation", {}).get("initial_train_days", 504)
        n_select = min(val_init, len(X))
        selected_features, _ = select_features_pipeline(
            X.iloc[:n_select], y.iloc[:n_select],
            max_features=max_features,
            corr_threshold=corr_threshold,
            method=fs_method,
        )
        X = X[selected_features]

    val_cfg = strategy.get("validation", {})
    validator = WalkForwardValidator(
        initial_train_days=val_cfg.get("initial_train_days", 504),
        test_days=val_cfg.get("test_days", 21),
        step_days=val_cfg.get("step_days", 21),
    )
    folds = validator.split(X)

    evaluator = MarketPulseEvaluator(num_classes=params["num_classes"])

    fold_accuracies = []
    fold_f1s = []
    fold_dates = []
    all_true = []
    all_pred = []

    progress = st.progress(0, text="Running walk-forward validation...")

    for i, fold in enumerate(folds):
        X_train, y_train, X_test, y_test = validator.get_fold_data(X, y, fold)

        model = _build_model(params)
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

    # Info bar
    model_label = (
        "Ensemble (XGB + LGB)" if params["use_ensemble"] else "XGBoost only"
    )
    st.info(
        f"**Strategy:** {params['strategy_name']} | "
        f"**Model:** {model_label} | "
        f"**Features:** technical + returns + adaptive + macro + regime | "
        f"**Folds:** {len(folds)}"
    )

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


def render_regression(params):
    """Regression analysis: predict return magnitude (Phase 4)."""
    st.header(f"📐 {params['ticker']} — Return Regression")

    fmt_ticker = params["market_config"].format_ticker(params["ticker"])

    df = load_enriched_data(
        fmt_ticker,
        params["start_date"],
        params["end_date"],
        params["market_name"],
        params["strategy_name"],
    )

    if df.empty:
        st.error(f"No data available for {fmt_ticker}")
        return

    # Generate regression labels
    labeled_df = generate_labels(
        df,
        horizon=params["horizon"],
        label_type="regression",
    )

    X, y = get_clean_features_and_labels(labeled_df)

    if len(X) < 600:
        st.warning(f"Insufficient data ({len(X)} samples).")
        return

    # Use regression strategy config if available
    try:
        reg_strategy = load_strategy_config("medium_term_regression")
    except Exception:
        reg_strategy = params["strategy_config"]

    val_cfg = reg_strategy.get("validation", {})
    validator = WalkForwardValidator(
        initial_train_days=val_cfg.get("initial_train_days", 504),
        test_days=val_cfg.get("test_days", 42),
        step_days=val_cfg.get("step_days", 21),
    )
    folds = validator.split(X)

    reg_evaluator = RegressionEvaluator()

    fold_results = []
    all_true = []
    all_pred = []

    progress = st.progress(0, text="Running regression walk-forward...")

    for i, fold in enumerate(folds):
        X_train, y_train, X_test, y_test = validator.get_fold_data(X, y, fold)

        ensemble = MarketPulseEnsemble.from_strategy_config(reg_strategy)
        ensemble.fit(X_train, y_train)

        y_pred_fold = ensemble.predict(X_test)

        result = reg_evaluator.evaluate_fold(
            y_true=y_test.values,
            y_pred=y_pred_fold,
            fold_number=fold.fold_number,
            train_size=fold.train_size,
            test_start_date=fold.test_start_date,
            test_end_date=fold.test_end_date,
        )
        fold_results.append(result)
        all_true.extend(y_test.values)
        all_pred.extend(y_pred_fold)

        progress.progress(
            (i + 1) / len(folds), text=f"Fold {i + 1}/{len(folds)}"
        )

    progress.empty()

    # Aggregate
    report = reg_evaluator.aggregate_results(
        fold_results, ticker=fmt_ticker, horizon=params["horizon"],
    )

    # Summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Directional Accuracy",
            f"{report.mean_directional_accuracy:.1%}",
        )
    with col2:
        st.metric("MAE", f"{report.mean_mae:.5f}")
    with col3:
        st.metric("R²", f"{report.mean_r2:.4f}")
    with col4:
        st.metric("Info. Coefficient", f"{report.mean_ic:.4f}")

    # Scatter plot
    st.subheader("Predicted vs Actual Returns")
    all_true_arr = np.array(all_true)
    all_pred_arr = np.array(all_pred)

    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(
        x=all_true_arr,
        y=all_pred_arr,
        mode="markers",
        marker=dict(size=4, color="steelblue", opacity=0.4),
        name="Predictions",
    ))
    lims = [
        min(all_true_arr.min(), all_pred_arr.min()),
        max(all_true_arr.max(), all_pred_arr.max()),
    ]
    fig_scatter.add_trace(go.Scatter(
        x=lims,
        y=lims,
        mode="lines",
        line=dict(color="red", dash="dash"),
        name="Perfect",
    ))
    fig_scatter.update_layout(
        title=f"Predicted vs Actual {params['horizon']}d Returns",
        xaxis_title="Actual Return",
        yaxis_title="Predicted Return",
        template="plotly_white",
        height=400,
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Directional accuracy per fold
    st.subheader("Directional Accuracy per Fold")
    das = [f.directional_accuracy for f in fold_results]
    fig_da = go.Figure()
    fig_da.add_trace(
        go.Bar(x=list(range(len(das))), y=das, marker_color="steelblue")
    )
    fig_da.add_hline(
        y=0.5,
        line_dash="dash",
        line_color="red",
        annotation_text="Random (50%)",
    )
    fig_da.add_hline(
        y=np.mean(das),
        line_dash="dash",
        line_color="green",
        annotation_text=f"Mean: {np.mean(das):.3f}",
    )
    fig_da.update_layout(
        xaxis_title="Fold",
        yaxis_title="Directional Accuracy",
        template="plotly_white",
        height=350,
    )
    st.plotly_chart(fig_da, use_container_width=True)


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
    elif view == "Regression":
        render_regression(params)
    elif view == "Clustering":
        render_clustering(params)


if __name__ == "__main__":
    main()
