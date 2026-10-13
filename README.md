# Wolt Demand Forecast

This project focuses on **demand forecasting for food delivery and retail operations**.  
The goal is to analyze historical sales data, identify demand patterns, and prepare a foundation for building forecasting models that support inventory planning, replenishment, and operations decisions.

## Project Goals

- Forecast future demand for stores and items
- Analyze overall sales trends and seasonality
- Capture weekly and monthly patterns
- Detect anomalies and data quality issues
- Support model development for business decision-making

## What This Project Includes

- Data loading and cleaning
- Exploratory Data Analysis (EDA)
- Time-based trend analysis
- Store-level and item-level analysis
- Anomaly detection
- Feature engineering for forecasting

## Main Data Fields

Typical fields in the dataset include:

- `date`: date of the record
- `store`: store identifier
- `item`: item identifier
- `sales`: sales volume

## Project Structure

```text
wolt-demand-forecast/
├── data/
│   └── raw/
├── notebooks/
│   └── notebook/
├── README.md
└── ...
```

## Requirements

- Python 3.10+
- pandas
- numpy
- matplotlib
- statsmodels
- jupyter

## Getting Started

```bash
git clone https://github.com/your-username/wolt-demand-forecast.git
cd wolt-demand-forecast
pip install -r requirements.txt
jupyter notebook
```

## How to Run

Open `notebooks/notebook/01_eda.ipynb` and run the cells in order to:

1. Load the data
2. Inspect basic statistics
3. Analyze trends and seasonality
4. Explore store/item patterns
5. Detect anomalies

## Notes

This project is designed to help understand demand behavior in food delivery and retail settings, and to provide a strong base for future forecasting models.

## Future Work

- Add more feature engineering
- Try machine learning and time series models
- Evaluate forecasting performance
- Visualize prediction results
- Incorporate holiday and promotion effects