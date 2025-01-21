import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import timedelta
import xgboost as xgb

try:
    from prophet import Prophet
except ImportError:
    from fbprophet import Prophet

# -------------------------------------------------------------------------
# 1. Helper Functions
# -------------------------------------------------------------------------

def load_and_clean_data(csv_path: str) -> pd.DataFrame:
    """
    Load the dataset and apply basic cleaning:
      - Convert 'Date' to datetime
      - Remove parentheses from 'Order_Demand' and convert to numeric
      - Drop invalid or negative demands
    Returns the cleaned DataFrame.
    """
    df = pd.read_csv(csv_path)
    # Convert Date
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    
    # Remove any non-numeric chars from Order_Demand
    df['Order_Demand'] = (
        df['Order_Demand']
        .astype(str)
        .str.replace(r'[^0-9.-]', '', regex=True)
    )
    df['Order_Demand'] = pd.to_numeric(df['Order_Demand'], errors='coerce')
    df = df.dropna(subset=['Order_Demand'])
    
    # Remove zero or negative if not valid
    df = df[df['Order_Demand'] > 0]
    
    return df


def group_data(df: pd.DataFrame, group_by: str) -> pd.DataFrame:
    """
    Group data by 'Date' + selected group_by dimension 
    (either 'Product_Code' or 'Product_Category' or none).
    For demonstration, we only forecast a single product/category at a time,
    so we will filter by a single 'Product_Code' or 'Product_Category'
    and then group by Date to get daily aggregated demand.
    """
    # For a single product or category, let's assume we've already filtered.
    # Now group by Date for daily total:
    df_grouped = df.groupby('Date')['Order_Demand'].sum().reset_index()
    # Sort
    df_grouped = df_grouped.sort_values('Date').reset_index(drop=True)
    return df_grouped


def cap_outliers(df: pd.DataFrame, col: str, percentile=0.99) -> pd.DataFrame:
    """
    Cap extreme outliers at a given percentile (e.g., 99th).
    """
    cap_value = df[col].quantile(percentile)
    df[col] = np.where(df[col] > cap_value, cap_value, df[col])
    return df


def reindex_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure there's a row for each calendar day from min to max in 'Date'."""
    df = df.sort_values('Date').reset_index(drop=True)
    all_days = pd.date_range(start=df['Date'].min(), end=df['Date'].max(), freq='D')
    df_indexed = df.set_index('Date').reindex(all_days, fill_value=0).reset_index()
    df_indexed.rename(columns={'index': 'Date'}, inplace=True)
    return df_indexed

def make_prophet_forecast(df: pd.DataFrame, forecast_horizon: int=90, use_log=False):
    # 1) Reindex daily
    df = reindex_daily(df)
    
    # 2) Train-test split
    cutoff_date = df['Date'].max() - pd.Timedelta(days=forecast_horizon)
    train = df[df['Date'] <= cutoff_date].copy()
    test = df[df['Date'] > cutoff_date].copy()
    
    if use_log:
        train_prophet = train.rename(columns={'Date': 'ds'})
        train_prophet['y'] = np.log1p(train_prophet['Order_Demand'])
        test_prophet = test.rename(columns={'Date': 'ds'})
        test_prophet['y'] = np.log1p(test_prophet['Order_Demand'])
        
        model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
        model.fit(train_prophet[['ds','y']])
        
        future = model.make_future_dataframe(periods=len(test_prophet), freq='D')
        forecast = model.predict(future)
        
        # Align forecast and test carefully
        fcst_test = forecast[forecast['ds'] > cutoff_date]
        
        # Invert log
        y_pred = np.expm1(fcst_test['yhat'].values)
        y_true = test['Order_Demand'].values
        
    else:
        # No log transform
        train_prophet = train.rename(columns={'Date': 'ds','Order_Demand': 'y'})
        test_prophet = test.rename(columns={'Date': 'ds','Order_Demand': 'y'})
        
        model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
        model.fit(train_prophet[['ds','y']])
        
        future = model.make_future_dataframe(periods=len(test_prophet), freq='D')
        forecast = model.predict(future)
        fcst_test = forecast[forecast['ds'] > cutoff_date]
        
        y_pred = fcst_test['yhat'].values
        y_true = test_prophet['y'].values
    
    # Double-check length
    if len(y_pred) != len(y_true):
        # Attempt to align by date
        fcst_test = fcst_test.set_index('ds')
        test_prophet = test_prophet.set_index('ds')
        joined = test_prophet.join(fcst_test, how='inner', rsuffix='_forecast')
        # In case there's a mismatch, now we align on index
        y_pred = joined['yhat'].values
        y_true = joined['y'].values
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    # If you have many zeros, MAPE can blow up. Use np.where() to avoid dividing by zero:
    y_true_safe = np.where(y_true == 0, 1, y_true)  # minimal fix
    mape = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100
    
    return {
        'forecast': forecast,
        'y_pred': y_pred,
        'y_true': y_true,
        'train_df': train,
        'test_df': test,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape
    }


def make_xgb_forecast(df: pd.DataFrame, forecast_horizon=90):
    """
    Train an XGBoost model with time-series features (lags, rolling means, date-based),
    then forecast the next 'forecast_horizon' days.

    Return predictions, plus metrics.
    """
    df = df.copy().sort_values('Date').reset_index(drop=True)
    
    # Train-test split by last 90 days
    cutoff_date = df['Date'].max() - pd.Timedelta(days=forecast_horizon)
    # We'll build features on the entire dataset first, then separate
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['day'] = df['Date'].dt.day
    df['month'] = df['Date'].dt.month
    df['year'] = df['Date'].dt.year
    
    # Lags
    df['lag1'] = df['Order_Demand'].shift(1)
    df['lag7'] = df['Order_Demand'].shift(7)
    df['lag14'] = df['Order_Demand'].shift(14)
    
    # Rolling means
    df['roll7_mean'] = df['Order_Demand'].rolling(7).mean()
    df['roll14_mean'] = df['Order_Demand'].rolling(14).mean()
    
    # Drop early rows with NaN from shifting
    df = df.dropna().reset_index(drop=True)
    
    # Now split
    train_xgb = df[df['Date'] <= cutoff_date]
    test_xgb = df[df['Date'] > cutoff_date]
    
    features = [
        'day_of_week','day','month','year',
        'lag1','lag7','lag14','roll7_mean','roll14_mean'
    ]
    
    X_train = train_xgb[features]
    y_train = train_xgb['Order_Demand']
    X_test = test_xgb[features]
    y_test = test_xgb['Order_Demand']
    
    model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    # If XGBoost version is older, remove eval_set/early_stopping_rounds
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test - y_pred)/(y_test+1e-9))) * 100
    
    return {
        'train_df': train_xgb,
        'test_df': test_xgb,
        'y_pred': y_pred,
        'y_true': y_test,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'model': model
    }


# -------------------------------------------------------------------------
# 2. Streamlit Application
# -------------------------------------------------------------------------
def main():
    st.title("Product Demand Forecasting - Advance Data Analytics")
    st.write("""
    **By Mali, 2025**  
    This application loads the historical product demand data, cleans it, and uses 
    advanced forecasting techniques (Prophet or XGBoost) to predict future demand.
    """)

    # 1) File upload or path
    st.sidebar.write("## Data Source")
    data_file = st.sidebar.file_uploader("Upload CSV file (Historical Product Demand)", type=["csv"])
    
    if not data_file:
        st.info("Please upload the 'Historical Product Demand.csv' to proceed.")
        return
    
    # 2) Load and clean data
    df = load_and_clean_data(data_file)
    st.write("### Basic Data Info")
    st.write("Initial shape: ", df.shape)
    st.write(df.head(5))
    
    # 3) Sidebar for selection
    st.sidebar.write("## Filtering Options")
    # Let user pick dimension to filter on
    dimension_choice = st.sidebar.selectbox("Forecast by:", ["Product_Code", "Product_Category"])
    
    # Unique values
    unique_vals = df[dimension_choice].unique().tolist()
    selection = st.sidebar.selectbox(f"Select {dimension_choice}:", unique_vals)
    
    # Filter
    df_sub = df[df[dimension_choice] == selection].copy()
    
    # Group by Date
    df_grouped = group_data(df_sub, dimension_choice)
    
    # Show chart of raw grouped data
    st.write(f"### Historical Daily Demand for {dimension_choice}: {selection}")
    if df_grouped.shape[0] == 0:
        st.warning(f"No data for {selection}. Please select a different {dimension_choice}.")
        return
    
    # Cap outliers
    df_grouped = cap_outliers(df_grouped, 'Order_Demand', percentile=0.99)
    
    fig_raw, ax_raw = plt.subplots(figsize=(10,4))
    ax_raw.plot(df_grouped['Date'], df_grouped['Order_Demand'], marker='o', linewidth=1)
    ax_raw.set_title(f"Daily Demand for {selection} (After Cleaning)")
    ax_raw.set_xlabel("Date")
    ax_raw.set_ylabel("Order_Demand")
    st.pyplot(fig_raw)
    
    # 4) Choose model
    st.sidebar.write("## Model Settings")
    model_choice = st.sidebar.radio("Which model do you want to use?", ["Prophet", "XGBoost"])
    
    horizon_days = st.sidebar.slider("Forecast Horizon (days)", min_value=30, max_value=180, value=60)
    
    # 5) Run forecast
    if model_choice == "Prophet":
        use_log = st.sidebar.checkbox("Use Log Transform?", value=True)
        result = make_prophet_forecast(df_grouped, forecast_horizon=horizon_days, use_log=use_log)
        
        # Display metrics
        st.write(f"### Prophet Results for {selection}")
        st.write(f"**MAE:**  {result['MAE']:.2f}")
        st.write(f"**RMSE:** {result['RMSE']:.2f}")
        st.write(f"**MAPE:** {result['MAPE']:.2f}%")
        
        # Plot test predictions vs actual
        test_df = result['test_df'].copy()
        test_dates = test_df['Date']
        y_true = result['y_true']
        y_pred = result['y_pred']
        
        fig_forecast, ax_fc = plt.subplots(figsize=(10,5))
        ax_fc.plot(test_dates, y_true, label="Actual", marker='o')
        ax_fc.plot(test_dates, y_pred, label="Prophet Forecast", marker='x')
        ax_fc.set_title(f"Prophet Forecast vs Actual ({selection})")
        ax_fc.set_xlabel("Date")
        ax_fc.set_ylabel("Order Demand")
        ax_fc.legend()
        st.pyplot(fig_forecast)
        
    else:  # "XGBoost"
        result_xgb = make_xgb_forecast(df_grouped, forecast_horizon=horizon_days)
        
        st.write(f"### XGBoost Results for {selection}")
        st.write(f"**MAE:**(Mean Average Error)  {result_xgb['MAE']:.2f}")
        st.write(f"**RMSE:** (Root Mean Squared Error) {result_xgb['RMSE']:.2f}")
        st.write(f"**MAPE:** (Mean Absolute Percentage Error) {result_xgb['MAPE']:.2f}%")
        
        # Plot
        test_df = result_xgb['test_df']
        test_dates = test_df['Date']
        y_true = result_xgb['y_true']
        y_pred = result_xgb['y_pred']
        
        fig_forecast, ax_fc = plt.subplots(figsize=(10,5))
        ax_fc.plot(test_dates, y_true, label="Actual", marker='o')
        ax_fc.plot(test_dates, y_pred, label="XGBoost Forecast", marker='^')
        ax_fc.set_title(f"XGBoost Forecast vs Actual ({selection})")
        ax_fc.set_xlabel("Date")
        ax_fc.set_ylabel("Order Demand")
        ax_fc.legend()
        st.pyplot(fig_forecast)
    
    st.write("**Done!**")

# Run the app
if __name__ == "__main__":
    main()
