# Urban Fuel Analytics Dashboard

A comprehensive Streamlit application for analysing consumer-survey data, building predictive models, and forecasting revenue for the Urban Fuel meal-kit venture.

## Features
1. **Data Visualisation** – 15+ interactive EDA charts with dynamic filters.
2. **Classification** – K-NN, Decision Tree, Random Forest, Gradient Boosting with full metrics, confusion matrix, ROC, batch prediction & download.
3. **Clustering** – K-Means with elbow chart, slider-controlled k, persona table, and CSV export.
4. **Association Rules** – Apriori on multi-select columns with configurable support & confidence.
5. **Regression & Forecast** – Linear, Ridge, Lasso, Decision-Tree regressors plus a 12-month revenue forecast by city.

## Setup
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
