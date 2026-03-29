# Washington, D.C. Bikeshare Demand Analysis

## Overview

This project is an end-to-end data analytics case study on Washington, D.C. bikeshare demand. It analyzes how ridership changes across time, weather, and rider type, then builds forecasting models to predict hourly demand. The repo now includes both a polished analysis notebook and a Streamlit dashboard so reviewers can either read the case study or interact with the results directly.

## Key Questions

- How does bikeshare ridership vary by hour, weekday structure, season, and weather?
- How do casual and registered riders differ in their usage patterns?
- Can hourly demand be predicted well enough to support planning and operations?

## Dataset

- Source: Capital Bikeshare hourly ridership dataset
- Time span: January 1, 2011 to December 31, 2012
- Granularity: hourly observations

The dataset includes calendar fields, weather conditions, and hourly counts for casual, registered, and total rides.

## Methods

- data cleaning and validation
- exploratory data analysis
- feature engineering for time, weather, and commute structure
- statistical comparisons with effect sizes and confidence intervals
- regression modeling with baseline and tree-based models
- out-of-sample evaluation using a chronological train/test split

## Key Findings

- Registered riders show strong commuting behavior, with the largest peaks on weekday rush hours.
- Casual riders are much more leisure-oriented and make up a larger share of demand on warm weekends and holidays.
- Weather matters operationally: ridership is meaningfully lower during precipitation than during clear conditions.
- Time-of-day is the strongest predictor of demand, but weather and day type add important forecasting signal.
- Demand increased noticeably from 2011 to 2012, suggesting stronger adoption over time.

## Modeling Summary

The main forecasting target is total hourly ridership, with supporting models for casual and registered riders. I compared a baseline mean, linear regression, ridge regression, random forest, and histogram gradient boosting using a chronological train/test split.

The best-performing model was histogram gradient boosting:

- Total rides: RMSE 65.35, MAE 40.56, R2 0.895
- Casual rides: RMSE 17.88, MAE 9.79, R2 0.863
- Registered rides: RMSE 56.61, MAE 34.75, R2 0.896

These results show that nonlinear models capture hourly demand patterns much better than simple baselines.

## Dashboard

The Streamlit dashboard turns the analysis into an interactive portfolio piece. It includes:

- a concise project intro and KPI summary row
- filterable views by season, weather, day type, month, and hour range
- interactive demand charts for time, weekday, season, weather, and rider behavior
- a modeling section with forecast metrics, feature importance, and actual-vs-predicted demand
- a scenario tool that predicts total hourly ridership for user-selected conditions

Main app file:

- `app.py`

## Limitations

- The dataset is observational, so the project does not make causal claims.
- There is no demographic data, so rider behavior cannot be analyzed by user characteristics.
- There is no station-level geographic detail in the main workflow.
- External drivers such as events and transit disruptions are not included.

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Regenerate outputs:

```bash
python src/generate_outputs.py
```

Open the notebook:

```bash
jupyter notebook notebooks/dc_bikeshare_case_study.ipynb
```

Run the dashboard:

```bash
streamlit run app.py
```

## Project Structure

- `notebooks/dc_bikeshare_case_study.ipynb`: main recruiter-facing case study notebook
- `app.py`: Streamlit dashboard for interactive portfolio review
- `src/bikeshare_analysis.py`: reusable analysis, feature engineering, modeling, and figure generation code
- `src/dashboard_utils.py`: dashboard-specific helpers for filtering, Plotly charts, and scenario prediction
- `outputs/figures/`: polished charts used in the notebook
- `outputs/tables/`: model metrics and supporting summary tables
- `data/processed/hourly_bikeshare_clean.csv`: cleaned analytical dataset
- `archive/`: legacy notebook and earlier artifacts retained for reference

## Portfolio Blurb

This project demonstrates end-to-end data analysis, from cleaning and exploratory analysis through predictive modeling and stakeholder communication. The notebook and dashboard together show how bikeshare usage data can be turned into clear operational insights, interactive exploration, and practical demand forecasts.
