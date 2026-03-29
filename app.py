from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))

from dashboard_utils import (
    METRIC_MAP,
    PALETTE,
    actual_vs_predicted_chart,
    bar_casual_share,
    bar_season_weather,
    bar_weekday_pattern,
    build_kpi_summary,
    derive_default_prediction_inputs,
    feature_importance_chart,
    filter_dashboard_data,
    line_daily_demand,
    line_hourly_profile,
    line_rider_mix,
    load_dashboard_data,
    load_feature_importance,
    load_model_metrics,
    make_prediction_row,
    model_comparison_chart,
    month_options,
    train_total_demand_model,
)


st.set_page_config(
    page_title="DC Bikeshare Demand Dashboard",
    page_icon="B",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles() -> None:
    st.markdown(
        f"""
        <style>
        .stApp {{
            background:
                radial-gradient(circle at top right, rgba(59,130,246,0.08), transparent 28%),
                linear-gradient(180deg, #f8fafc 0%, #eef4f7 100%);
        }}
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
        }}
        [data-testid="stSidebar"] * {{
            color: #e5eefb;
        }}
        .hero {{
            padding: 1.25rem 0 1.6rem 0;
            border-bottom: 1px solid rgba(15, 23, 42, 0.08);
            margin-bottom: 1rem;
        }}
        .eyebrow {{
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            color: {PALETTE["teal"]};
            font-weight: 700;
            margin-bottom: 0.4rem;
        }}
        .hero h1 {{
            font-size: 3rem;
            line-height: 1.02;
            color: {PALETTE["ink"]};
            margin: 0 0 0.65rem 0;
        }}
        .hero p {{
            font-size: 1rem;
            max-width: 52rem;
            color: #334155;
            margin: 0;
        }}
        .note {{
            background: rgba(255,255,255,0.8);
            border: 1px solid rgba(148,163,184,0.25);
            border-radius: 18px;
            padding: 0.95rem 1rem;
            color: #334155;
        }}
        .section-label {{
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            color: {PALETTE["slate"]};
            font-weight: 700;
            margin-top: 0.8rem;
        }}
        .section-title {{
            font-size: 1.55rem;
            color: {PALETTE["ink"]};
            margin: 0.1rem 0 0.25rem 0;
            font-weight: 700;
        }}
        .section-copy {{
            color: #475569;
            max-width: 48rem;
            margin-bottom: 1rem;
        }}
        .metric-shell {{
            background: rgba(255,255,255,0.88);
            border: 1px solid rgba(148,163,184,0.2);
            border-radius: 18px;
            padding: 1rem 1rem 0.9rem 1rem;
            min-height: 118px;
            box-shadow: 0 12px 32px rgba(15, 23, 42, 0.05);
        }}
        .metric-label {{
            color: #64748b;
            font-size: 0.83rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.4rem;
        }}
        .metric-value {{
            color: {PALETTE["ink"]};
            font-size: 1.85rem;
            font-weight: 700;
            line-height: 1.1;
        }}
        .insight-box {{
            background: rgba(255,255,255,0.9);
            border: 1px solid rgba(148,163,184,0.18);
            border-radius: 20px;
            padding: 1rem 1.15rem;
            min-height: 120px;
        }}
        .insight-box b {{
            color: {PALETTE["ink"]};
        }}
        .caption {{
            color: #64748b;
            font-size: 0.92rem;
            margin-top: -0.25rem;
            margin-bottom: 1rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def section_header(label: str, title: str, copy: str) -> None:
    st.markdown(f'<div class="section-label">{label}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-copy">{copy}</div>', unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def get_data() -> pd.DataFrame:
    return load_dashboard_data()


@st.cache_data(show_spinner=False)
def get_metrics_table() -> pd.DataFrame:
    return load_model_metrics()


@st.cache_data(show_spinner=False)
def get_importance_table() -> pd.DataFrame:
    return load_feature_importance()


@st.cache_resource(show_spinner=False)
def get_total_model(df: pd.DataFrame):
    return train_total_demand_model(df)


def render_metric_card(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="metric-shell">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    inject_styles()
    bike = get_data()
    metrics = get_metrics_table()
    importance = get_importance_table()
    model, heldout_predictions = get_total_model(bike)

    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">Portfolio Dashboard</div>
            <h1>Washington, D.C. Bikeshare Demand</h1>
            <p>
                Explore how ridership changes across time, weather, and rider type, then inspect a practical demand
                forecast built from calendar and weather signals. The dashboard is designed for fast portfolio review:
                what changed, what matters, and how well demand can be predicted.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    intro_cols = st.columns([1.2, 1])
    with intro_cols[0]:
        st.markdown(
            """
            **Why it matters:** Bikeshare operators need to anticipate commute peaks, weather-driven drops, and shifts
            between casual and registered riders to plan inventory, rebalancing, and staffing.
            """
        )
    with intro_cols[1]:
        st.markdown(
            """
            <div class="note"><b>How to use this dashboard:</b> filter the historical data in the sidebar, scan the KPI row,
            then move through demand patterns, rider behavior, and the forecasting section.</div>
            """,
            unsafe_allow_html=True,
        )

    st.sidebar.markdown("## Explore the data")
    demand_label = st.sidebar.selectbox("Demand metric", list(METRIC_MAP.keys()), index=0)
    metric_col = METRIC_MAP[demand_label]

    season_options = sorted(bike["season_label"].dropna().unique().tolist())
    weather_options = sorted(bike["weather_group"].dropna().unique().tolist())
    day_type_options = sorted(bike["day_type"].dropna().unique().tolist())
    month_names = month_options()

    seasons = st.sidebar.multiselect("Season", season_options, default=season_options)
    weather_groups = st.sidebar.multiselect("Weather", weather_options, default=weather_options)
    day_types = st.sidebar.multiselect("Day type", day_type_options, default=day_type_options)
    months = st.sidebar.multiselect("Month", month_names, default=month_names)
    hour_range = st.sidebar.slider("Hour range", 0, 23, (0, 23))
    holiday_mode = st.sidebar.selectbox("Holiday filter", ["All hours", "Exclude holidays", "Holiday only"], index=0)

    filtered = filter_dashboard_data(
        bike,
        seasons=seasons,
        weather_groups=weather_groups,
        day_types=day_types,
        months=months,
        hour_range=hour_range,
        holiday_mode=holiday_mode,
    )

    if filtered.empty:
        st.warning("The current filter combination returns no rows. Widen the filters to continue exploring demand.")
        return

    section_header(
        "Overview",
        "Selected KPIs",
        "These metrics update with the active filters so reviewers can quickly see the scale, mix, and strongest demand windows.",
    )
    kpis = build_kpi_summary(filtered)
    cols = st.columns(6)
    for col, (label, value) in zip(cols, kpis.items()):
        with col:
            render_metric_card(label, value)

    section_header(
        "Demand Patterns",
        "How ridership changes across time and conditions",
        f"The charts below focus on **{demand_label.lower()}** under the current filter set.",
    )
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(line_daily_demand(filtered, metric_col), use_container_width=True)
    with c2:
        st.plotly_chart(line_hourly_profile(filtered, metric_col), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(bar_weekday_pattern(filtered, metric_col), use_container_width=True)
    with c4:
        st.plotly_chart(bar_season_weather(filtered, metric_col), use_container_width=True)

    section_header(
        "Rider Behavior",
        "Where casual and registered riders diverge",
        "This view keeps the full rider split visible so a reviewer can see the commute-versus-leisure story clearly.",
    )
    c5, c6 = st.columns(2)
    with c5:
        st.plotly_chart(line_rider_mix(filtered), use_container_width=True)
    with c6:
        st.plotly_chart(bar_casual_share(filtered), use_container_width=True)

    rider_notes = st.columns(3)
    notes = [
        "Registered riders dominate weekday peak hours, which points to a strong commuting use case.",
        "Casual riders take a larger share on weekends, holidays, and in better weather, suggesting more discretionary trips.",
        "Weather reduces total demand, but the rider mix still changes with trip purpose and day structure.",
    ]
    for col, note in zip(rider_notes, notes):
        with col:
            st.markdown(f'<div class="insight-box">{note}</div>', unsafe_allow_html=True)

    section_header(
        "Modeling",
        "A practical demand forecast for total hourly rides",
        "The forecasting view focuses on total ridership because it is the most operationally useful target for planning inventory and staffing.",
    )

    mc1, mc2 = st.columns([1.05, 0.95])
    with mc1:
        st.plotly_chart(model_comparison_chart(metrics), use_container_width=True)
        st.markdown(
            '<div class="caption">Models compared: baseline mean, linear regression, ridge regression, random forest, and histogram gradient boosting.</div>',
            unsafe_allow_html=True,
        )
    with mc2:
        total_metrics = metrics.loc[metrics["target"] == "cnt"].sort_values("RMSE").reset_index(drop=True)
        st.dataframe(
            total_metrics[["model", "MAE", "RMSE", "R2"]].round(3),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown(
            """
            **Features used**

            Calendar context, hour of day, working-day status, weather conditions, and engineered cyclic time features.
            """
        )

    st.plotly_chart(actual_vs_predicted_chart(heldout_predictions), use_container_width=True)

    model_cols = st.columns([1.2, 0.8])
    with model_cols[0]:
        st.plotly_chart(feature_importance_chart(importance), use_container_width=True)
    with model_cols[1]:
        best_row = metrics.loc[metrics["target"] == "cnt"].sort_values("RMSE").iloc[0]
        st.markdown(
            f"""
            <div class="insight-box">
                <b>What this performance means in practice</b><br><br>
                The best model is <b>{best_row["model"]}</b>, with RMSE <b>{best_row["RMSE"]:.2f}</b> and
                MAE <b>{best_row["MAE"]:.2f}</b> on the held-out quarter of 2012. That is strong enough to support
                directional planning, but not precise enough to treat as a real-time operational control system.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Try a forecast scenario")
    defaults = derive_default_prediction_inputs(bike)
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        date_value = st.date_input("Date", value=defaults["date_value"])
        hour = st.slider("Hour", 0, 23, int(defaults["hour"]))
        season_label = st.selectbox("Season", ["Winter", "Spring", "Summer", "Fall"], index=defaults["season_code"] - 1)
    with p2:
        month_name = st.selectbox("Month", month_names, index=defaults["month_code"] - 1)
        weekday_name = st.selectbox(
            "Weekday",
            ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"],
            index=int(defaults["weekday"]),
        )
        weather_label = st.selectbox(
            "Weather",
            ["Clear / Few clouds", "Mist / Cloudy", "Light rain or snow", "Heavy rain or snow"],
            index=defaults["weather_code"] - 1,
        )
    with p3:
        holiday = st.selectbox("Holiday", ["No", "Yes"], index=int(defaults["holiday"]))
        workingday = st.selectbox("Working day", ["No", "Yes"], index=int(defaults["workingday"]))
        temp_c = st.slider("Temperature (C)", 0.0, 41.0, float(defaults["temp_c"]), 0.5)
    with p4:
        feels_like_c = st.slider("Feels like (C)", 0.0, 50.0, float(defaults["feels_like_c"]), 0.5)
        humidity_pct = st.slider("Humidity (%)", 0.0, 100.0, float(defaults["humidity_pct"]), 1.0)
        windspeed_kph = st.slider("Windspeed (kph)", 0.0, 57.0, float(defaults["windspeed_kph"]), 0.5)

    season_code_map = {"Winter": 1, "Spring": 2, "Summer": 3, "Fall": 4}
    weather_code_map = {
        "Clear / Few clouds": 1,
        "Mist / Cloudy": 2,
        "Light rain or snow": 3,
        "Heavy rain or snow": 4,
    }
    weekday_code_map = {
        "Sunday": 0,
        "Monday": 1,
        "Tuesday": 2,
        "Wednesday": 3,
        "Thursday": 4,
        "Friday": 5,
        "Saturday": 6,
    }
    month_code = [k for k, v in {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}.items() if v == month_name][0]
    prediction_row = make_prediction_row(
        date_value=pd.Timestamp(date_value),
        season_code=season_code_map[season_label],
        month_code=month_code,
        hour=hour,
        holiday=1 if holiday == "Yes" else 0,
        weekday=weekday_code_map[weekday_name],
        workingday=1 if workingday == "Yes" else 0,
        weather_code=weather_code_map[weather_label],
        temp_c=temp_c,
        feels_like_c=feels_like_c,
        humidity_pct=humidity_pct,
        windspeed_kph=windspeed_kph,
    )
    predicted_rides = float(model.predict(prediction_row)[0])

    st.success(f"Predicted total hourly ridership: {predicted_rides:,.0f} rides")
    st.caption("This forecast is based on historical patterns from 2011-2012 and should be interpreted as a scenario estimate, not a causal or real-time prediction.")

    section_header(
        "Takeaways",
        "Key insights for a fast portfolio review",
        "These are the main conclusions a recruiter or stakeholder should leave with after a quick scan.",
    )
    insight_cols = st.columns(2)
    insight_text = [
        "Registered riders behave like commuters: weekday rush hours create the strongest demand peaks in the dataset.",
        "Casual riders are more sensitive to weather and weekends, so rider mix changes meaningfully as conditions improve.",
        "Time-of-day is the most powerful forecast signal, but weather and working-day structure add important lift.",
        "Demand drops under precipitation, making weather severity a practical planning feature rather than just descriptive context.",
        "A nonlinear boosting model materially improves on linear baselines, which shows demand is driven by interactions rather than one simple linear trend.",
        "The project is strongest as an operational demand case study, not as a causal or equity analysis, because key demographic and geographic detail is missing.",
    ]
    for col, text in zip(insight_cols * 3, insight_text):
        with col:
            st.markdown(f'<div class="insight-box">{text}</div>', unsafe_allow_html=True)

    section_header(
        "Limits",
        "What this dashboard does not claim",
        "The app is designed to be transparent about scope so the project feels credible rather than overextended.",
    )
    st.markdown(
        """
        - The dataset is historical and observational, so the dashboard does not make causal claims.
        - There is no rider demographic information, so equity and user-profile questions cannot be answered directly.
        - The main workflow does not include station-level geography, which limits local rebalancing and neighborhood analysis.
        - External drivers such as special events, transit disruptions, and tourism likely explain some remaining forecast error.
        """
    )


if __name__ == "__main__":
    main()
