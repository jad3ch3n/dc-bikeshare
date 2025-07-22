# 🚲 Bikeshare Usage Analysis – Washington D.C.

This project explores usage patterns in Washington D.C.’s bike-sharing system using hourly and daily data from 2011 to 2012. I investigate behavioral trends across different rider types (casual vs. registered), identify temporal and weather-based patterns, and reflect on the limitations of the dataset from an equity standpoint.

---

## 📊 Project Summary

- **User Behavior:**  
  Registered users tend to ride during weekday commuting hours, while casual users prefer weekends and warmer days.

- **Temporal Trends:**  
  Hourly usage shows clear rush-hour peaks for registered users. Kernel density estimates visualize different riding habits across working and non-working days.

- **Temperature Impact:**  
  LOWESS-smoothed plots show a strong positive relationship between temperature and ridership, especially for casual users.

- **Equity Considerations:**  
  The dataset lacks demographic or geographic identifiers, limiting insight into equity. Proposed improvements include collecting user age, gender, income, and neighborhood.

- **Expansion Proposal:**  
  Cities with warm climates and existing transit infrastructure, such as San Francisco, Atlanta, and Los Angeles, are recommended for system expansion.

---

## 🧰 Tools & Technologies

- Python
- Jupyter Notebook
- Pandas, Seaborn, Matplotlib
- Statsmodels (LOWESS smoothing)
- Markdown

---

## 📁 Files Included

- `dc-bikeshare.ipynb` – Full exploratory notebook
- `dc-bikeshare-paper.html` – Final project writeup with visuals and analysis (web-ready)
- `.png` visualizations – Plots generated throughout the analysis
- `bikeshare.txt` – Dataset metadata
- `ds100_utils.py` – Utility functions used in the notebook

---

## 🖼️ Sample Visuals

![Bivariate KDE Plot](images/Bivariate_KDE_Plot_Comparison_of_Registered_vs_Casual_Riders_Legend_by_Working_and_Non_Working_Days.png)
*Bivariate KDE Plot Comparison of Registered vs Casual Riders (Legend by Working and Non-Working Days)*

![LOWESS Curves](images/LOWESS_Curves_Temperature_vs_Casual_Rider_Proportion_by_Day_of_the_Week.png)
*LOWESS Curves: Temperature vs. Casual Rider Proportion by Day of the Week*

---

## 🚀 Run It Yourself

### Option 1: Locally
```bash
pip install notebook pandas seaborn matplotlib statsmodels ipython
jupyter notebook dc-bikeshare.ipynb
```

### Option 2: In the Cloud
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jad3ch3n/dc-bikeshare/blob/main/dc-bikeshare.ipynb)

---

## ✍️ Author

**Jade Chen**  
[Portfolio](https://jad3ch3n.github.io/) | [LinkedIn](https://www.linkedin.com/in/jad3ch3n) | [GitHub](https://github.com/jad3ch3n)
