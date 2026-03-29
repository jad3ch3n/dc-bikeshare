from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline

from bikeshare_analysis import (
    MODEL_FEATURES,
    MONTH_MAP,
    NUMERIC_FEATURES,
    RAW_DATA_PATH,
    build_daily_summary,
    build_preprocessor,
    clean_bikeshare_data,
    load_raw_data,
    split_train_test,
)


ROOT_DIR = Path(__file__).resolve().parent.parent
TABLE_DIR = ROOT_DIR / "outputs" / "tables"

PALETTE = {
    "ink": "#0f172a",
    "blue": "#1d4ed8",
    "teal": "#0f766e",
    "orange": "#ea580c",
    "slate": "#64748b",
    "cloud": "#e2e8f0",
}

METRIC_MAP = {
    "Total rides": "cnt",
    "Casual rides": "casual",
    "Registered rides": "registered",
}


def load_dashboard_data() -> pd.DataFrame:
    processed_path = ROOT_DIR / "data" / "processed" / "hourly_bikeshare_clean.csv"
    if processed_path.exists():
        df = pd.read_csv(processed_path)
        df["dteday"] = pd.to_datetime(df["dteday"])
        df["date_hour"] = pd.to_datetime(df["date_hour"])
        return df
    return clean_bikeshare_data(load_raw_data(RAW_DATA_PATH))


def filter_dashboard_data(
    df: pd.DataFrame,
    seasons: list[str],
    weather_groups: list[str],
    day_types: list[str],
    months: list[str],
    hour_range: tuple[int, int],
    holiday_mode: str,
) -> pd.DataFrame:
    filtered = df.copy()
    filtered = filtered.loc[filtered["season_label"].isin(seasons)]
    filtered = filtered.loc[filtered["weather_group"].isin(weather_groups)]
    filtered = filtered.loc[filtered["day_type"].isin(day_types)]
    filtered = filtered.loc[filtered["month_name"].isin(months)]
    filtered = filtered.loc[filtered["hr"].between(hour_range[0], hour_range[1])]

    if holiday_mode == "Holiday only":
        filtered = filtered.loc[filtered["holiday"].eq(1)]
    elif holiday_mode == "Exclude holidays":
        filtered = filtered.loc[filtered["holiday"].eq(0)]

    return filtered


def build_kpi_summary(df: pd.DataFrame) -> dict[str, str]:
    daily = build_daily_summary(df)
    peak_hour = (
        df.groupby("hr", as_index=False)["cnt"].mean().sort_values("cnt", ascending=False).iloc[0]
    )
    peak_season = (
        df.groupby("season_label", as_index=False)["cnt"].mean().sort_values("cnt", ascending=False).iloc[0]
    )
    return {
        "Total rides": f"{int(df['cnt'].sum()):,}",
        "Average hourly demand": f"{df['cnt'].mean():.1f}",
        "Average daily demand": f"{daily['total_rides'].mean():.0f}",
        "Casual share": f"{df['casual'].sum() / df['cnt'].sum():.1%}",
        "Peak hour": f"{int(peak_hour['hr']):02d}:00",
        "Peak season": str(peak_season["season_label"]),
    }


def load_model_metrics() -> pd.DataFrame:
    metrics = pd.read_csv(TABLE_DIR / "model_metrics.csv")
    target_labels = {"cnt": "Total rides", "casual": "Casual rides", "registered": "Registered rides"}
    metrics["target_label"] = metrics["target"].map(target_labels)
    return metrics


def load_feature_importance() -> pd.DataFrame:
    return pd.read_csv(TABLE_DIR / "total_ridership_feature_importance.csv")


def train_total_demand_model(df: pd.DataFrame) -> tuple[Pipeline, pd.DataFrame]:
    train_df, test_df = split_train_test(df)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "model",
                HistGradientBoostingRegressor(
                    random_state=42,
                    max_depth=8,
                    learning_rate=0.08,
                    max_iter=300,
                ),
            ),
        ]
    )
    pipeline.fit(train_df[MODEL_FEATURES], train_df["cnt"])
    predicted = pipeline.predict(test_df[MODEL_FEATURES])

    comparison = test_df[["dteday", "hr", "cnt", "season_label", "weather_group"]].copy()
    comparison["predicted_cnt"] = predicted
    comparison["date_hour"] = test_df["date_hour"].values
    return pipeline, comparison


def make_prediction_row(
    date_value: pd.Timestamp,
    season_code: int,
    month_code: int,
    hour: int,
    holiday: int,
    weekday: int,
    workingday: int,
    weather_code: int,
    temp_c: float,
    feels_like_c: float,
    humidity_pct: float,
    windspeed_kph: float,
) -> pd.DataFrame:
    raw = pd.DataFrame(
        [
            {
                "instant": 0,
                "dteday": pd.Timestamp(date_value).date().isoformat(),
                "season": season_code,
                "yr": int(pd.Timestamp(date_value).year - 2011),
                "mnth": month_code,
                "hr": hour,
                "holiday": holiday,
                "weekday": weekday,
                "workingday": workingday,
                "weathersit": weather_code,
                "temp": temp_c / 41,
                "atemp": feels_like_c / 50,
                "hum": humidity_pct / 100,
                "windspeed": windspeed_kph / 67,
                "casual": 0,
                "registered": 0,
                "cnt": 0,
            }
        ]
    )
    engineered = clean_bikeshare_data(raw)
    return engineered[MODEL_FEATURES]


def theme_figure(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="white",
        font={"family": "Arial", "color": PALETTE["ink"]},
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        hoverlabel={"bgcolor": "white"},
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.18)", zeroline=False)
    return fig


def line_daily_demand(df: pd.DataFrame, metric_col: str) -> go.Figure:
    daily = build_daily_summary(df)
    value_col = {
        "cnt": "total_rides",
        "casual": "casual_rides",
        "registered": "registered_rides",
    }[metric_col]
    daily["rolling_avg"] = daily[value_col].rolling(30, min_periods=7).mean()
    fig = px.line(
        daily,
        x="dteday",
        y=[value_col, "rolling_avg"],
        color_discrete_sequence=[PALETTE["cloud"], PALETTE["blue"]],
        labels={"value": "Daily rides", "dteday": "Date", "variable": ""},
    )
    fig.update_traces(line={"width": 2.5})
    fig.update_layout(title="Ridership over time", legend={"orientation": "h", "y": 1.08, "x": 0})
    return theme_figure(fig)


def line_hourly_profile(df: pd.DataFrame, metric_col: str) -> go.Figure:
    hourly = df.groupby(["day_type", "hr"], as_index=False)[metric_col].mean()
    fig = px.line(
        hourly,
        x="hr",
        y=metric_col,
        color="day_type",
        markers=True,
        color_discrete_sequence=[PALETTE["blue"], PALETTE["orange"]],
        labels={"hr": "Hour of day", metric_col: "Average hourly rides", "day_type": ""},
        title="Hourly demand pattern",
    )
    return theme_figure(fig)


def bar_weekday_pattern(df: pd.DataFrame, metric_col: str) -> go.Figure:
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday = df.groupby("weekday_name", as_index=False)[metric_col].mean()
    weekday["weekday_name"] = pd.Categorical(weekday["weekday_name"], categories=order, ordered=True)
    weekday = weekday.sort_values("weekday_name")
    fig = px.bar(
        weekday,
        x="weekday_name",
        y=metric_col,
        color=metric_col,
        color_continuous_scale=["#dbeafe", "#60a5fa", "#1d4ed8"],
        labels={"weekday_name": "", metric_col: "Average hourly rides"},
        title="Average demand by weekday",
    )
    fig.update_layout(coloraxis_showscale=False)
    return theme_figure(fig)


def bar_season_weather(df: pd.DataFrame, metric_col: str) -> go.Figure:
    summary = (
        df.groupby(["season_label", "weather_group"], as_index=False)[metric_col]
        .mean()
        .sort_values(metric_col, ascending=False)
    )
    fig = px.bar(
        summary,
        x="season_label",
        y=metric_col,
        color="weather_group",
        barmode="group",
        color_discrete_sequence=[PALETTE["blue"], PALETTE["teal"], PALETTE["orange"]],
        labels={"season_label": "", metric_col: "Average hourly rides", "weather_group": "Weather"},
        title="Demand by season and weather",
    )
    return theme_figure(fig)


def line_rider_mix(df: pd.DataFrame) -> go.Figure:
    hourly = (
        df.groupby("hr", as_index=False)[["casual", "registered"]]
        .mean()
        .melt(id_vars="hr", var_name="rider_type", value_name="avg_rides")
    )
    fig = px.line(
        hourly,
        x="hr",
        y="avg_rides",
        color="rider_type",
        markers=True,
        color_discrete_map={"casual": PALETTE["orange"], "registered": PALETTE["blue"]},
        labels={"hr": "Hour of day", "avg_rides": "Average hourly rides", "rider_type": ""},
        title="Casual vs. registered rider behavior",
    )
    return theme_figure(fig)


def bar_casual_share(df: pd.DataFrame) -> go.Figure:
    summary = (
        df.groupby(["weather_group", "day_type"], as_index=False)["casual_share"]
        .mean()
        .sort_values(["weather_group", "day_type"])
    )
    fig = px.bar(
        summary,
        x="weather_group",
        y="casual_share",
        color="day_type",
        barmode="group",
        color_discrete_sequence=[PALETTE["blue"], PALETTE["orange"]],
        labels={"weather_group": "Weather", "casual_share": "Casual share", "day_type": ""},
        title="Casual riders take a larger share on leisure-oriented days",
    )
    fig.update_yaxes(tickformat=".0%")
    return theme_figure(fig)


def model_comparison_chart(metrics: pd.DataFrame) -> go.Figure:
    total_only = metrics.loc[metrics["target"] == "cnt"].sort_values("RMSE", ascending=True)
    fig = px.bar(
        total_only,
        x="RMSE",
        y="model",
        orientation="h",
        color="model",
        color_discrete_sequence=[PALETTE["slate"], "#60a5fa", PALETTE["blue"], "#9a6b46", PALETTE["teal"]],
        labels={"RMSE": "Test RMSE", "model": ""},
        title="Forecast comparison for total hourly demand",
    )
    fig.update_layout(showlegend=False)
    return theme_figure(fig)


def actual_vs_predicted_chart(prediction_df: pd.DataFrame) -> go.Figure:
    sample = prediction_df.copy().sort_values("date_hour").iloc[:500]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sample["date_hour"],
            y=sample["cnt"],
            mode="lines",
            name="Actual",
            line={"color": PALETTE["blue"], "width": 2},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sample["date_hour"],
            y=sample["predicted_cnt"],
            mode="lines",
            name="Predicted",
            line={"color": PALETTE["orange"], "width": 2},
        )
    )
    fig.update_layout(title="Actual vs. predicted demand on the held-out period")
    fig.update_yaxes(title="Hourly rides")
    fig.update_xaxes(title="Date")
    return theme_figure(fig)


def feature_importance_chart(importance_df: pd.DataFrame) -> go.Figure:
    top = importance_df.head(8).sort_values("importance", ascending=True)
    fig = px.bar(
        top,
        x="importance",
        y="feature",
        orientation="h",
        color="importance",
        color_continuous_scale=["#d1fae5", "#14b8a6", "#0f766e"],
        labels={"importance": "Permutation importance", "feature": ""},
        title="Most useful forecast features",
    )
    fig.update_layout(coloraxis_showscale=False)
    return theme_figure(fig)


def derive_default_prediction_inputs(df: pd.DataFrame) -> dict[str, object]:
    mode_row = df.mode(numeric_only=False).iloc[0]
    return {
        "date_value": pd.Timestamp("2012-10-15"),
        "season_code": int(mode_row["season"]),
        "month_code": 10,
        "hour": 8,
        "holiday": 0,
        "weekday": 1,
        "workingday": 1,
        "weather_code": int(mode_row["weathersit"]),
        "temp_c": round(float(df["temp_c"].median()), 1),
        "feels_like_c": round(float(df["feels_like_c"].median()), 1),
        "humidity_pct": round(float(df["humidity_pct"].median()), 1),
        "windspeed_kph": round(float(df["windspeed_kph"].median()), 1),
    }


def month_options() -> list[str]:
    return [MONTH_MAP[i] for i in sorted(MONTH_MAP)]
