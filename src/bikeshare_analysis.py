from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = ROOT_DIR / "data" / "data.zip"
PROCESSED_DATA_PATH = ROOT_DIR / "data" / "processed" / "hourly_bikeshare_clean.csv"
FIGURE_DIR = ROOT_DIR / "outputs" / "figures"
TABLE_DIR = ROOT_DIR / "outputs" / "tables"
TRAIN_END_DATE = pd.Timestamp("2012-09-30")

SEASON_MAP = {1: "Winter", 2: "Spring", 3: "Summer", 4: "Fall"}
WEATHER_MAP = {
    1: "Clear / Few clouds",
    2: "Mist / Cloudy",
    3: "Light rain or snow",
    4: "Heavy rain or snow",
}
WEATHER_GROUP_MAP = {
    1: "Clear",
    2: "Cloudy / mist",
    3: "Precipitation",
    4: "Precipitation",
}
WEEKDAY_MAP = {
    0: "Sunday",
    1: "Monday",
    2: "Tuesday",
    3: "Wednesday",
    4: "Thursday",
    5: "Friday",
    6: "Saturday",
}
MONTH_MAP = {
    1: "Jan",
    2: "Feb",
    3: "Mar",
    4: "Apr",
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Aug",
    9: "Sep",
    10: "Oct",
    11: "Nov",
    12: "Dec",
}

TARGETS = ["cnt", "casual", "registered"]
NUMERIC_FEATURES = [
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed",
    "is_weekend",
    "is_rush_hour",
    "is_commute_window",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "feels_temp_gap",
    "weather_severity",
]
CAT_FEATURES = [
    "season_label",
    "weather_label",
    "weather_group",
    "weekday_name",
    "day_type",
    "temp_band",
    "hour_band",
]
MODEL_FEATURES = NUMERIC_FEATURES + CAT_FEATURES


@dataclass
class ModelArtifact:
    target: str
    model_name: str
    pipeline: Pipeline | None
    y_true: pd.Series
    y_pred: np.ndarray


def ensure_output_dirs() -> None:
    for path in [PROCESSED_DATA_PATH.parent, FIGURE_DIR, TABLE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def set_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_bikeshare_data(df: pd.DataFrame) -> pd.DataFrame:
    bike = df.copy()
    bike["dteday"] = pd.to_datetime(bike["dteday"])
    bike = bike.sort_values(["dteday", "hr"]).reset_index(drop=True)

    if (bike["casual"] + bike["registered"]).ne(bike["cnt"]).any():
        raise ValueError("Total ridership does not match casual + registered for every row.")

    bike["season_label"] = bike["season"].map(SEASON_MAP)
    bike["weather_label"] = bike["weathersit"].map(WEATHER_MAP)
    bike["weather_group"] = bike["weathersit"].map(WEATHER_GROUP_MAP)
    bike["weekday_name"] = bike["weekday"].map(WEEKDAY_MAP)
    bike["month_name"] = bike["mnth"].map(MONTH_MAP)

    bike["date_hour"] = bike["dteday"] + pd.to_timedelta(bike["hr"], unit="h")
    bike["year"] = 2011 + bike["yr"]
    bike["temp_c"] = bike["temp"] * 41
    bike["feels_like_c"] = bike["atemp"] * 50
    bike["humidity_pct"] = bike["hum"] * 100
    bike["windspeed_kph"] = bike["windspeed"] * 67
    bike["casual_share"] = bike["casual"] / bike["cnt"]

    bike["is_weekend"] = bike["weekday"].isin([0, 6]).astype(int)
    bike["is_rush_hour"] = bike["hr"].isin([7, 8, 9, 16, 17, 18]).astype(int)
    bike["is_commute_window"] = bike["hr"].isin([6, 7, 8, 9, 16, 17, 18, 19]).astype(int)
    bike["day_type"] = np.where(bike["workingday"].eq(1), "Working day", "Weekend / holiday")

    bike["hour_band"] = pd.cut(
        bike["hr"],
        bins=[-1, 5, 9, 15, 19, 23],
        labels=["Overnight", "Morning commute", "Midday", "Evening commute", "Late evening"],
    )
    bike["temp_band"] = pd.cut(
        bike["temp_c"],
        bins=[-1, 10, 20, 30, 50],
        labels=["Cold (<10C)", "Mild (10-20C)", "Warm (20-30C)", "Hot (30C+)"],
    )
    bike["weather_severity"] = bike["weathersit"].replace({4: 3})
    bike["feels_temp_gap"] = bike["atemp"] - bike["temp"]
    bike["hour_sin"] = np.sin(2 * np.pi * bike["hr"] / 24)
    bike["hour_cos"] = np.cos(2 * np.pi * bike["hr"] / 24)
    bike["month_sin"] = np.sin(2 * np.pi * (bike["mnth"] - 1) / 12)
    bike["month_cos"] = np.cos(2 * np.pi * (bike["mnth"] - 1) / 12)

    return bike


def save_processed_data(df: pd.DataFrame) -> None:
    ensure_output_dirs()
    df.to_csv(PROCESSED_DATA_PATH, index=False)


def build_data_quality_table(df: pd.DataFrame) -> pd.DataFrame:
    expected_rows = (df["dteday"].max() - df["dteday"].min()).days + 1
    observed_hours = df["hr"].value_counts().sort_index()

    summary = pd.DataFrame(
        [
            ("Rows", int(len(df)), "Hourly observations across 2011-2012"),
            ("Date range", f"{df['dteday'].min().date()} to {df['dteday'].max().date()}", "Two full calendar years"),
            ("Missing values", int(df.isna().sum().sum()), "No null values detected"),
            ("Duplicate rows", int(df.duplicated().sum()), "No exact duplicates detected"),
            (
                "Target consistency",
                bool((df["casual"] + df["registered"]).eq(df["cnt"]).all()),
                "Total ridership equals casual + registered in every row",
            ),
            ("Distinct days", int(df["dteday"].nunique()), f"Expected {expected_rows} days in range"),
            (
                "Hours observed",
                f"{int(observed_hours.min())}-{int(observed_hours.max())} rows per hour",
                "A small number of timestamps are absent in the source data",
            ),
        ],
        columns=["check", "result", "interpretation"],
    )
    return summary


def build_feature_dictionary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("season_label", "Season category", "Captures broad annual seasonality"),
            ("weather_group", "Collapsed weather severity", "Separates clear, cloudy, and precipitation conditions"),
            ("day_type", "Working day vs weekend / holiday", "Distinguishes commute demand from leisure demand"),
            ("hour_band", "Grouped hour buckets", "Improves readability for stakeholder summaries"),
            ("temp_band", "Grouped temperature levels", "Captures nonlinear weather comfort effects"),
            ("hour_sin / hour_cos", "Cyclical hour encoding", "Preserves circular nature of time-of-day"),
            ("month_sin / month_cos", "Cyclical month encoding", "Preserves seasonality without artificial jumps"),
            ("is_rush_hour", "AM / PM commute indicator", "Highlights peak commuting windows"),
            ("is_commute_window", "Broader commute proxy", "Captures surrounding shoulder periods"),
            ("feels_temp_gap", "atemp - temp", "Proxy for perceived vs measured conditions"),
        ],
        columns=["feature", "type", "why_it_matters"],
    )


def build_daily_summary(df: pd.DataFrame) -> pd.DataFrame:
    daily = (
        df.groupby("dteday", as_index=False)
        .agg(
            total_rides=("cnt", "sum"),
            casual_rides=("casual", "sum"),
            registered_rides=("registered", "sum"),
            mean_temp_c=("temp_c", "mean"),
            mean_humidity_pct=("humidity_pct", "mean"),
            mean_windspeed_kph=("windspeed_kph", "mean"),
            workingday=("workingday", "max"),
            holiday=("holiday", "max"),
            dominant_weather=("weather_group", lambda s: s.mode().iat[0]),
            season=("season_label", "first"),
        )
    )
    daily["casual_share"] = daily["casual_rides"] / daily["total_rides"]
    return daily


def bootstrap_mean_difference(
    sample_a: pd.Series,
    sample_b: pd.Series,
    label_a: str,
    label_b: str,
    metric_name: str,
) -> dict[str, Any]:
    sample_a = sample_a.dropna()
    sample_b = sample_b.dropna()
    diff = sample_a.mean() - sample_b.mean()
    pooled_std = np.sqrt((sample_a.var(ddof=1) + sample_b.var(ddof=1)) / 2)
    effect_size = diff / pooled_std if pooled_std else np.nan
    ci = stats.bootstrap(
        (sample_a.to_numpy(), sample_b.to_numpy()),
        lambda a, b: np.mean(a) - np.mean(b),
        confidence_level=0.95,
        n_resamples=5000,
        random_state=42,
        method="basic",
    ).confidence_interval
    return {
        "comparison": f"{label_a} minus {label_b}",
        "metric": metric_name,
        "mean_a": sample_a.mean(),
        "mean_b": sample_b.mean(),
        "difference": diff,
        "ci_low": ci.low,
        "ci_high": ci.high,
        "cohens_d": effect_size,
        "mann_whitney_pvalue": stats.mannwhitneyu(sample_a, sample_b, alternative="two-sided").pvalue,
    }


def compute_statistical_insights(df: pd.DataFrame) -> pd.DataFrame:
    comparisons = [
        bootstrap_mean_difference(
            df.loc[df["is_weekend"].eq(1), "casual"],
            df.loc[df["is_weekend"].eq(0), "casual"],
            "Weekend casual demand",
            "Weekday casual demand",
            "Hourly casual riders",
        ),
        bootstrap_mean_difference(
            df.loc[df["workingday"].eq(1) & df["is_rush_hour"].eq(1), "registered"],
            df.loc[df["workingday"].eq(0) & df["is_rush_hour"].eq(1), "registered"],
            "Working-day rush-hour registered demand",
            "Non-working-day rush-hour registered demand",
            "Hourly registered riders",
        ),
        bootstrap_mean_difference(
            df.loc[df["weathersit"].eq(1), "cnt"],
            df.loc[df["weathersit"].ge(3), "cnt"],
            "Clear-weather demand",
            "Precipitation demand",
            "Hourly total riders",
        ),
    ]
    return pd.DataFrame(comparisons)


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CAT_FEATURES,
            ),
        ]
    )


def build_model_catalog() -> dict[str, Any]:
    return {
        "Baseline mean": None,
        "Linear regression": LinearRegression(),
        "Ridge regression": Ridge(alpha=2.0),
        "Random forest": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "Histogram gradient boosting": HistGradientBoostingRegressor(
            random_state=42,
            max_depth=8,
            learning_rate=0.08,
            max_iter=300,
        ),
    }


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "R2": r2_score(y_true, y_pred),
    }


def split_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = df.loc[df["dteday"] <= TRAIN_END_DATE].copy()
    test_df = df.loc[df["dteday"] > TRAIN_END_DATE].copy()
    return train_df, test_df


def run_modeling(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, ModelArtifact], pd.DataFrame, pd.DataFrame]:
    train_df, test_df = split_train_test(df)
    X_train = train_df[MODEL_FEATURES]
    X_test = test_df[MODEL_FEATURES]

    preprocessor = build_preprocessor()
    models = build_model_catalog()

    metric_rows: list[dict[str, Any]] = []
    best_artifacts: dict[str, ModelArtifact] = {}

    for target in TARGETS:
        y_train = train_df[target]
        y_test = test_df[target]

        best_rmse = np.inf
        for model_name, estimator in models.items():
            if estimator is None:
                predictions = np.repeat(y_train.mean(), len(y_test))
                artifact = ModelArtifact(
                    target=target,
                    model_name=model_name,
                    pipeline=None,
                    y_true=y_test,
                    y_pred=predictions,
                )
            else:
                pipeline = Pipeline(
                    steps=[
                        ("preprocessor", preprocessor),
                        ("model", estimator),
                    ]
                )
                pipeline.fit(X_train, y_train)
                predictions = pipeline.predict(X_test)
                artifact = ModelArtifact(
                    target=target,
                    model_name=model_name,
                    pipeline=pipeline,
                    y_true=y_test,
                    y_pred=predictions,
                )

            metrics = regression_metrics(y_test, predictions)
            metric_rows.append(
                {
                    "target": target,
                    "model": model_name,
                    "MAE": metrics["MAE"],
                    "RMSE": metrics["RMSE"],
                    "R2": metrics["R2"],
                }
            )

            if metrics["RMSE"] < best_rmse:
                best_rmse = metrics["RMSE"]
                best_artifacts[target] = artifact

    metrics_df = pd.DataFrame(metric_rows).sort_values(["target", "RMSE"]).reset_index(drop=True)
    importance_df = compute_best_model_importance(df, best_artifacts["cnt"])
    coefficient_df = compute_ridge_coefficients(df)
    return metrics_df, best_artifacts, importance_df, coefficient_df


def compute_best_model_importance(df: pd.DataFrame, artifact: ModelArtifact) -> pd.DataFrame:
    if artifact.pipeline is None:
        return pd.DataFrame(columns=["feature", "importance"])

    _, test_df = split_train_test(df)
    X_test = test_df[MODEL_FEATURES]
    y_test = test_df[artifact.target]

    importance = permutation_importance(
        artifact.pipeline,
        X_test,
        y_test,
        n_repeats=10,
        random_state=42,
        scoring="neg_root_mean_squared_error",
    )
    return (
        pd.DataFrame({"feature": MODEL_FEATURES, "importance": importance.importances_mean})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def compute_ridge_coefficients(df: pd.DataFrame) -> pd.DataFrame:
    train_df, _ = split_train_test(df)
    X_train = train_df[MODEL_FEATURES]
    y_train = train_df["cnt"]

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", Ridge(alpha=2.0)),
        ]
    )
    pipeline.fit(X_train, y_train)

    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    coefficients = pipeline.named_steps["model"].coef_
    coef_df = pd.DataFrame({"feature": feature_names, "coefficient": coefficients})
    coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
    return coef_df.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)


def save_tables(
    quality_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    importance_df: pd.DataFrame,
    coefficient_df: pd.DataFrame,
) -> None:
    ensure_output_dirs()
    quality_df.to_csv(TABLE_DIR / "data_quality_summary.csv", index=False)
    feature_df.to_csv(TABLE_DIR / "feature_dictionary.csv", index=False)
    stats_df.to_csv(TABLE_DIR / "statistical_comparisons.csv", index=False)
    metrics_df.to_csv(TABLE_DIR / "model_metrics.csv", index=False)
    importance_df.to_csv(TABLE_DIR / "total_ridership_feature_importance.csv", index=False)
    coefficient_df.to_csv(TABLE_DIR / "ridge_coefficients_total_ridership.csv", index=False)


def plot_daily_demand(df: pd.DataFrame) -> plt.Figure:
    daily = build_daily_summary(df)
    daily["rolling_total"] = daily["total_rides"].rolling(30, min_periods=7).mean()

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(daily["dteday"], daily["total_rides"], color="#9dbbd5", linewidth=1.2, alpha=0.6, label="Daily total rides")
    ax.plot(daily["dteday"], daily["rolling_total"], color="#0b5d7a", linewidth=3, label="30-day rolling average")
    ax.set_title("Washington, D.C. bikeshare demand rose sharply from 2011 to 2012")
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily rides")
    ax.legend(frameon=False)
    return fig


def plot_target_distribution(df: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.histplot(df["cnt"], bins=40, color="#0b5d7a", ax=axes[0])
    axes[0].set_title("Hourly ridership is right-skewed with a long high-demand tail")
    axes[0].set_xlabel("Total hourly rides")
    axes[0].set_ylabel("Number of hours")

    sample = df[["casual", "registered"]].rename(
        columns={"casual": "Casual riders", "registered": "Registered riders"}
    )
    sns.boxplot(data=sample, palette=["#f28e2b", "#1f77b4"], ax=axes[1])
    axes[1].set_title("Registered demand is consistently higher than casual demand")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Hourly riders")

    fig.suptitle("Ridership volume varies substantially across hours and rider segments", y=1.03)
    return fig


def plot_hourly_profiles(df: pd.DataFrame) -> plt.Figure:
    hourly = (
        df.groupby(["day_type", "hr"], as_index=False)
        .agg(casual=("casual", "mean"), registered=("registered", "mean"))
        .melt(id_vars=["day_type", "hr"], var_name="rider_type", value_name="mean_riders")
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    palette = {"casual": "#f28e2b", "registered": "#1f77b4"}

    for ax, day_type in zip(axes, ["Working day", "Weekend / holiday"]):
        subset = hourly.loc[hourly["day_type"] == day_type]
        sns.lineplot(
            data=subset,
            x="hr",
            y="mean_riders",
            hue="rider_type",
            palette=palette,
            linewidth=3,
            ax=ax,
        )
        ax.set_title(day_type)
        ax.set_xlabel("Hour of day")
        ax.set_ylabel("Average hourly riders")
        ax.legend(title="", frameon=False)

    fig.suptitle("Registered riders dominate commute peaks while casual demand skews later and leisure-oriented", y=1.03)
    return fig


def plot_weekday_hour_heatmap(df: pd.DataFrame) -> plt.Figure:
    heatmap_data = df.pivot_table(index="weekday_name", columns="hr", values="cnt", aggfunc="mean")
    heatmap_data = heatmap_data.reindex(
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    )

    fig, ax = plt.subplots(figsize=(16, 6))
    sns.heatmap(heatmap_data, cmap="YlGnBu", linewidths=0.4, cbar_kws={"label": "Average hourly rides"}, ax=ax)
    ax.set_title("Demand concentrates around weekday commute windows and weekend afternoons")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("")
    return fig


def plot_weather_impact(df: pd.DataFrame) -> plt.Figure:
    weather_summary = (
        df.groupby(["weather_group", "day_type"], as_index=False)
        .agg(mean_total_rides=("cnt", "mean"), mean_casual_share=("casual_share", "mean"))
    )
    weather_order = ["Clear", "Cloudy / mist", "Precipitation"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.barplot(
        data=weather_summary,
        x="weather_group",
        y="mean_total_rides",
        hue="day_type",
        order=weather_order,
        palette=["#0b5d7a", "#f28e2b"],
        ax=axes[0],
    )
    axes[0].set_title("Demand falls as weather worsens")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Average hourly rides")
    axes[0].legend(title="", frameon=False)

    sns.barplot(
        data=weather_summary,
        x="weather_group",
        y="mean_casual_share",
        hue="day_type",
        order=weather_order,
        palette=["#0b5d7a", "#f28e2b"],
        ax=axes[1],
    )
    axes[1].set_title("Casual share stays highest on leisure-oriented days")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Casual share of rides")
    axes[1].yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    axes[1].legend_.remove()

    fig.suptitle("Weather suppresses volume, but rider mix still depends heavily on trip purpose", y=1.03)
    return fig


def plot_temperature_mix(df: pd.DataFrame) -> plt.Figure:
    temp_summary = (
        df.groupby(["temp_band", "day_type"], observed=False, as_index=False)
        .agg(mean_casual_share=("casual_share", "mean"), mean_total_rides=("cnt", "mean"))
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    order = ["Cold (<10C)", "Mild (10-20C)", "Warm (20-30C)", "Hot (30C+)"]

    sns.lineplot(
        data=temp_summary,
        x="temp_band",
        y="mean_total_rides",
        hue="day_type",
        marker="o",
        linewidth=3,
        palette=["#0b5d7a", "#f28e2b"],
        sort=False,
        ax=axes[0],
    )
    axes[0].set_title("Warmer weather lifts overall ridership")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Average hourly rides")

    sns.lineplot(
        data=temp_summary,
        x="temp_band",
        y="mean_casual_share",
        hue="day_type",
        marker="o",
        linewidth=3,
        palette=["#0b5d7a", "#f28e2b"],
        sort=False,
        ax=axes[1],
    )
    axes[1].set_title("Casual riders become a larger share as conditions improve")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Casual share of rides")
    axes[1].yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    axes[1].legend(title="", frameon=False)
    axes[0].legend_.remove()

    for ax in axes:
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=15, ha="right")

    fig.suptitle("Temperature affects both demand volume and the balance between rider segments", y=1.05)
    return fig


def plot_model_comparison(metrics_df: pd.DataFrame) -> plt.Figure:
    plot_df = metrics_df.copy()
    target_names = {"cnt": "Total rides", "casual": "Casual rides", "registered": "Registered rides"}
    plot_df["target_label"] = plot_df["target"].map(target_names)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=False)
    palette = {
        "Baseline mean": "#b8c4cc",
        "Linear regression": "#86b4c6",
        "Ridge regression": "#3a7ca5",
        "Random forest": "#9c6644",
        "Histogram gradient boosting": "#2a9d8f",
    }

    for ax, target in zip(axes, TARGETS):
        subset = plot_df.loc[plot_df["target"] == target].sort_values("RMSE", ascending=True)
        sns.barplot(
            data=subset,
            y="model",
            x="RMSE",
            hue="model",
            palette=palette,
            dodge=False,
            legend=False,
            ax=ax,
        )
        ax.set_title(target_names[target])
        ax.set_xlabel("Test RMSE")
        ax.set_ylabel("")

    fig.suptitle("A tree-based model materially improves forecast accuracy over linear baselines", y=1.03)
    return fig


def plot_actual_vs_predicted(artifact: ModelArtifact) -> plt.Figure:
    residuals = artifact.y_true.to_numpy() - artifact.y_pred

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].scatter(artifact.y_true, artifact.y_pred, alpha=0.35, color="#0b5d7a", edgecolor="none")
    lims = [0, max(artifact.y_true.max(), artifact.y_pred.max())]
    axes[0].plot(lims, lims, "--", color="#f28e2b", linewidth=2)
    axes[0].set_title(f"Actual vs predicted total rides\n{artifact.model_name}")
    axes[0].set_xlabel("Actual rides")
    axes[0].set_ylabel("Predicted rides")

    axes[1].scatter(artifact.y_pred, residuals, alpha=0.35, color="#2a9d8f", edgecolor="none")
    axes[1].axhline(0, linestyle="--", color="#f28e2b", linewidth=2)
    axes[1].set_title("Residuals remain wider on the busiest hours")
    axes[1].set_xlabel("Predicted rides")
    axes[1].set_ylabel("Residual (actual - predicted)")

    return fig


def save_figure(fig: plt.Figure, filename: str) -> None:
    ensure_output_dirs()
    fig.savefig(FIGURE_DIR / filename, dpi=180)
    plt.close(fig)


def generate_figures(df: pd.DataFrame, metrics_df: pd.DataFrame, best_artifacts: dict[str, ModelArtifact]) -> None:
    set_plot_style()
    figures = {
        "01_daily_demand_trend.png": plot_daily_demand(df),
        "02_target_distribution.png": plot_target_distribution(df),
        "03_hourly_rider_profiles.png": plot_hourly_profiles(df),
        "04_weekday_hour_heatmap.png": plot_weekday_hour_heatmap(df),
        "05_weather_impact.png": plot_weather_impact(df),
        "06_temperature_mix.png": plot_temperature_mix(df),
        "07_model_comparison.png": plot_model_comparison(metrics_df),
        "08_actual_vs_predicted_total_rides.png": plot_actual_vs_predicted(best_artifacts["cnt"]),
    }
    for filename, fig in figures.items():
        save_figure(fig, filename)


def generate_project_outputs() -> dict[str, pd.DataFrame]:
    ensure_output_dirs()
    cleaned_df = clean_bikeshare_data(load_raw_data())
    save_processed_data(cleaned_df)

    quality_df = build_data_quality_table(cleaned_df)
    feature_df = build_feature_dictionary()
    stats_df = compute_statistical_insights(cleaned_df)
    metrics_df, best_artifacts, importance_df, coefficient_df = run_modeling(cleaned_df)

    save_tables(quality_df, feature_df, stats_df, metrics_df, importance_df, coefficient_df)
    generate_figures(cleaned_df, metrics_df, best_artifacts)

    return {
        "cleaned_df": cleaned_df,
        "quality_df": quality_df,
        "feature_df": feature_df,
        "stats_df": stats_df,
        "metrics_df": metrics_df,
        "importance_df": importance_df,
        "coefficient_df": coefficient_df,
    }


if __name__ == "__main__":
    generate_project_outputs()
