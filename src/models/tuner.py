"""
Hyperparameter tuning for MarketPulse models.

Supports:
- Optuna (Bayesian / TPE) — recommended, fewest trials for best result
- Random search — quick baseline
- Grid search — exhaustive but slow

All methods use walk-forward validation to avoid look-ahead bias.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from ..data.fetcher import create_fetcher
from ..data.market_config import load_market_config, load_strategy_config
from ..data.preprocessing import preprocess_ohlcv
from ..features.labels import generate_labels, get_clean_features_and_labels
from ..features.returns import compute_return_features
from ..features.technical import compute_technical_indicators
from ..utils.validation import WalkForwardValidator
from .xgboost_classifier import MarketPulseXGBClassifier

logger = logging.getLogger(__name__)

# ────────────────────── Search Spaces ──────────────────────

XGBOOST_SEARCH_SPACE = {
    "max_depth": {"type": "int", "low": 3, "high": 10},
    "n_estimators": {"type": "int", "low": 100, "high": 800, "step": 50},
    "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
    "subsample": {"type": "float", "low": 0.5, "high": 1.0},
    "colsample_bytree": {"type": "float", "low": 0.4, "high": 1.0},
    "min_child_weight": {"type": "int", "low": 1, "high": 10},
    "gamma": {"type": "float", "low": 0.0, "high": 1.0},
    "reg_alpha": {"type": "float", "low": 1e-3, "high": 10.0, "log": True},
    "reg_lambda": {"type": "float", "low": 1e-3, "high": 10.0, "log": True},
}

LIGHTGBM_SEARCH_SPACE = {
    "max_depth": {"type": "int", "low": 3, "high": 10},
    "n_estimators": {"type": "int", "low": 100, "high": 800, "step": 50},
    "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
    "subsample": {"type": "float", "low": 0.5, "high": 1.0},
    "colsample_bytree": {"type": "float", "low": 0.4, "high": 1.0},
    "min_child_weight": {"type": "int", "low": 1, "high": 10},
    "reg_alpha": {"type": "float", "low": 1e-3, "high": 10.0, "log": True},
    "reg_lambda": {"type": "float", "low": 1e-3, "high": 10.0, "log": True},
    "num_leaves": {"type": "int", "low": 15, "high": 127},
}


def _sample_param(trial, name: str, spec: dict) -> Any:
    """Sample a single parameter from an Optuna trial."""
    if spec["type"] == "int":
        return trial.suggest_int(
            name, spec["low"], spec["high"], step=spec.get("step", 1)
        )
    elif spec["type"] == "float":
        if spec.get("log", False):
            return trial.suggest_float(name, spec["low"], spec["high"], log=True)
        return trial.suggest_float(name, spec["low"], spec["high"])
    elif spec["type"] == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    else:
        raise ValueError(f"Unknown param type: {spec['type']}")


# ────────────────────── Tuner Class ──────────────────────


class MarketPulseTuner:
    """Hyperparameter optimizer for MarketPulse models.

    Uses walk-forward cross-validation as the objective, ensuring
    no data leakage during tuning.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (aligned, NaN-free).
    y : pd.Series
        Labels.
    strategy_config : dict
        Strategy configuration.
    search_space : dict, optional
        Param name → {type, low, high, ...}. Defaults to XGBOOST_SEARCH_SPACE.
    metric : str
        Metric to optimize: 'f1_macro', 'accuracy', 'balanced_accuracy'.
    max_folds : int, optional
        Limit number of walk-forward folds per trial for speed.
    selected_features : list, optional
        If provided, only use these features during tuning.
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        strategy_config: dict,
        search_space: Optional[dict] = None,
        metric: str = "f1_macro",
        max_folds: Optional[int] = None,
        selected_features: Optional[List[str]] = None,
    ):
        self.X = X[selected_features] if selected_features else X
        self.y = y
        self.strategy_config = strategy_config
        self.search_space = search_space or XGBOOST_SEARCH_SPACE
        self.metric = metric
        self.max_folds = max_folds

        self.num_classes = strategy_config.get("num_classes", 3)

        # Set up walk-forward validator
        self.validator = WalkForwardValidator.from_strategy_config(strategy_config)

        # Results storage
        self.best_params: Dict[str, Any] = {}
        self.best_score: float = 0.0
        self.study = None  # Optuna study object
        self.all_trials: List[Dict] = []

    def _evaluate_params(self, hyperparams: Dict) -> float:
        """Evaluate a parameter set using walk-forward CV.

        Returns
        -------
        float
            Mean metric across folds.
        """
        folds = self.validator.split(self.X)

        if self.max_folds:
            # Use the most recent folds (most relevant)
            folds = folds[-self.max_folds:]

        scores = []
        for fold in folds:
            X_train, y_train, X_test, y_test = self.validator.get_fold_data(
                self.X, self.y, fold
            )

            model = MarketPulseXGBClassifier(
                hyperparameters=hyperparams,
                num_classes=self.num_classes,
            )
            model.fit(X_train, y_train, balance_classes=True)

            y_pred = model.predict(X_test)
            y_true = y_test.values.astype(int)

            if self.metric == "f1_macro":
                score = f1_score(y_true, y_pred, average="macro", zero_division=0)
            elif self.metric == "accuracy":
                score = (y_true == y_pred).mean()
            elif self.metric == "balanced_accuracy":
                from sklearn.metrics import balanced_accuracy_score
                score = balanced_accuracy_score(y_true, y_pred)
            else:
                raise ValueError(f"Unknown metric: {self.metric}")

            scores.append(score)

        return float(np.mean(scores))

    def tune_optuna(
        self,
        n_trials: int = 50,
        timeout: Optional[int] = None,
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        """Run Bayesian optimization with Optuna (TPE sampler).

        Parameters
        ----------
        n_trials : int
            Number of parameter combinations to try.
        timeout : int, optional
            Maximum time in seconds.
        show_progress : bool
            Whether to show Optuna's progress bar.

        Returns
        -------
        dict
            Best hyperparameters found.
        """
        try:
            import optuna
        except ImportError:
            raise ImportError(
                "Optuna required for Bayesian optimization. "
                "Install with: pip install optuna"
            )

        # Suppress Optuna's default output when not needed
        if not show_progress:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            params = {}
            for name, spec in self.search_space.items():
                params[name] = _sample_param(trial, name, spec)
            params["random_state"] = 42
            params["n_jobs"] = -1
            params["verbosity"] = 0

            score = self._evaluate_params(params)
            return score

        self.study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(),
        )

        self.study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=show_progress,
        )

        self.best_params = self.study.best_params
        self.best_params["random_state"] = 42
        self.best_params["n_jobs"] = -1
        self.best_params["verbosity"] = 0
        self.best_score = self.study.best_value

        # Store all trials
        self.all_trials = [
            {
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "state": str(t.state),
            }
            for t in self.study.trials
        ]

        logger.info(
            f"Optuna tuning complete: best {self.metric}={self.best_score:.4f} "
            f"in {len(self.study.trials)} trials"
        )
        logger.info(f"Best params: {self.best_params}")

        return self.best_params

    def tune_random(
        self,
        n_trials: int = 30,
    ) -> Dict[str, Any]:
        """Random search over the parameter space.

        Parameters
        ----------
        n_trials : int
            Number of random combinations to evaluate.

        Returns
        -------
        dict
            Best hyperparameters found.
        """
        rng = np.random.RandomState(42)

        best_score = -1.0
        best_params = {}

        for i in range(n_trials):
            params = {}
            for name, spec in self.search_space.items():
                if spec["type"] == "int":
                    step = spec.get("step", 1)
                    vals = range(spec["low"], spec["high"] + 1, step)
                    params[name] = rng.choice(list(vals))
                elif spec["type"] == "float":
                    if spec.get("log", False):
                        params[name] = float(
                            np.exp(rng.uniform(np.log(spec["low"]), np.log(spec["high"])))
                        )
                    else:
                        params[name] = float(rng.uniform(spec["low"], spec["high"]))
                elif spec["type"] == "categorical":
                    params[name] = rng.choice(spec["choices"])

            params["random_state"] = 42
            params["n_jobs"] = -1
            params["verbosity"] = 0

            score = self._evaluate_params(params)

            logger.info(f"  Trial {i + 1}/{n_trials}: {self.metric}={score:.4f}")

            self.all_trials.append({
                "number": i,
                "value": score,
                "params": params,
                "state": "COMPLETE",
            })

            if score > best_score:
                best_score = score
                best_params = params.copy()
                logger.info(f"  ★ New best: {self.metric}={score:.4f}")

        self.best_params = best_params
        self.best_score = best_score

        logger.info(
            f"Random search complete: best {self.metric}={self.best_score:.4f}"
        )
        return self.best_params

    def get_results_df(self) -> pd.DataFrame:
        """Get all trial results as a DataFrame."""
        if not self.all_trials:
            return pd.DataFrame()

        rows = []
        for trial in self.all_trials:
            row = {"trial": trial["number"], "score": trial["value"]}
            row.update(trial["params"])
            rows.append(row)

        return pd.DataFrame(rows).sort_values("score", ascending=False)

    def get_param_importance(self) -> Optional[pd.Series]:
        """Get hyperparameter importance from Optuna study.

        Only available after tune_optuna().
        """
        if self.study is None:
            logger.warning("No Optuna study available. Run tune_optuna() first.")
            return None

        try:
            import optuna
            importance = optuna.importance.get_param_importances(self.study)
            return pd.Series(importance).sort_values(ascending=False)
        except Exception as e:
            logger.warning(f"Could not compute param importance: {e}")
            return None


# ────────────────────── Convenience Function ──────────────────────


def quick_tune(
    ticker: str = "AAPL",
    market: str = "stocks",
    strategy: str = "short_term",
    method: str = "optuna",
    n_trials: int = 30,
    max_folds: int = 10,
    metric: str = "f1_macro",
) -> Tuple[Dict, float, pd.DataFrame]:
    """One-liner to tune hyperparameters for a single ticker.

    Parameters
    ----------
    ticker : str
        Ticker symbol.
    market : str
        Market config name.
    strategy : str
        Strategy config name.
    method : str
        'optuna' or 'random'.
    n_trials : int
        Number of trials.
    max_folds : int
        Use only the N most recent walk-forward folds (speed up).
    metric : str
        Optimization metric.

    Returns
    -------
    Tuple[dict, float, pd.DataFrame]
        (best_params, best_score, all_trials_df)

    Example
    -------
    >>> best_params, score, trials = quick_tune("AAPL", n_trials=30)
    >>> print(f"Best F1: {score:.4f}")
    >>> print(best_params)
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    market_config = load_market_config(market)
    strategy_config = load_strategy_config(strategy)

    fmt_ticker = market_config.format_ticker(ticker)
    logger.info(f"Tuning {fmt_ticker} with {method} ({n_trials} trials)...")

    # Fetch & preprocess
    from ..data.fetcher import create_fetcher
    fetcher = create_fetcher(market_config=market_config, source=market_config.data_source)
    years = strategy_config.get("data", {}).get("years_of_history", 5)
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=years * 365)).strftime("%Y-%m-%d")

    raw_df = fetcher.fetch(fmt_ticker, start=start_date, end=end_date)
    df = preprocess_ohlcv(raw_df, market_config=market_config)
    df = compute_technical_indicators(df)
    df = compute_return_features(df)

    horizon = strategy_config.get("default_horizon", 1)
    df = generate_labels(
        df,
        horizon=horizon,
        label_type=strategy_config.get("label_type", "classification"),
        num_classes=strategy_config.get("num_classes", 3),
        threshold=strategy_config.get("threshold", 0.01),
    )
    X, y = get_clean_features_and_labels(df)

    # Feature selection (optional, from config)
    fs_config = strategy_config.get("feature_selection", {})
    selected_features = None
    if fs_config.get("method", "none") != "none":
        from ..features.feature_selection import select_features_pipeline
        sel_names, _ = select_features_pipeline(
            X, y,
            max_features=fs_config.get("max_features", 15),
            method=fs_config.get("method", "importance"),
        )
        selected_features = sel_names

    # Tune
    tuner = MarketPulseTuner(
        X=X,
        y=y,
        strategy_config=strategy_config,
        metric=metric,
        max_folds=max_folds,
        selected_features=selected_features,
    )

    if method == "optuna":
        best_params = tuner.tune_optuna(n_trials=n_trials)
    else:
        best_params = tuner.tune_random(n_trials=n_trials)

    return best_params, tuner.best_score, tuner.get_results_df()


# ────────────────────── CLI ──────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="MarketPulse Hyperparameter Tuner")
    parser.add_argument("--ticker", default="AAPL", help="Ticker to tune on")
    parser.add_argument("--market", default="stocks")
    parser.add_argument("--strategy", default="short_term")
    parser.add_argument("--method", default="optuna", choices=["optuna", "random"])
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--folds", type=int, default=10, help="Max walk-forward folds")
    parser.add_argument("--metric", default="f1_macro")

    args = parser.parse_args()

    best_params, best_score, trials_df = quick_tune(
        ticker=args.ticker,
        market=args.market,
        strategy=args.strategy,
        method=args.method,
        n_trials=args.trials,
        max_folds=args.folds,
        metric=args.metric,
    )

    print(f"\n{'=' * 50}")
    print(f"Best {args.metric}: {best_score:.4f}")
    print(f"Best params:")
    for k, v in sorted(best_params.items()):
        if k not in ("random_state", "n_jobs", "verbosity"):
            print(f"  {k}: {v}")
    print(f"\nTop 5 trials:")
    print(trials_df.head(5)[["trial", "score"]].to_string(index=False))


if __name__ == "__main__":
    main()
