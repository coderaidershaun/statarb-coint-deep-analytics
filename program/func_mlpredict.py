from constants import TICKER_1, TICKER_2, WINDOW
from func_stats import calculate_cointegration_metrics
from func_exchange import get_current_dydx_data
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
import math


# Make ML Predictions
def make_predictions():
  df = get_current_dydx_data()
  series_1 = df[f"{TICKER_1}_close"].astype(float)
  series_2 = df[f"{TICKER_2}_close"].astype(float)
  periods = WINDOW
  spreads = [math.nan] * periods

  # Get Metrics
  print("Gathering rolling metrics...")
  for i in range(len(df) - periods):

    # Construct Series
    series_1 = df[f"{TICKER_1}_close"].astype(float)[i:i+periods]
    series_2 = df[f"{TICKER_2}_close"].astype(float)[i:i+periods]

    # Get Rolling Spread
    p_value, t_check, z_score, spread = calculate_cointegration_metrics(series_1, series_2)
    spreads.append(spread)

  # Add spreads
  df["spread"] = spreads

  # Calculate Returns
  df["ticker_1_rets"] = df[f"{TICKER_1}_close"].pct_change()
  df["ticker_2_rets"] = df[f"{TICKER_2}_close"].pct_change()

  # Drop and deal with NA or inf values
  df.dropna(inplace=True)
  df.replace([np.inf, -np.inf], np.nan, inplace=True)

  # Extract ACTUAL
  df["ACTUAL"] = 0
  df.loc[df["spread"].shift(-1) > df["spread"], "ACTUAL"] = 1
  actuals = df["ACTUAL"].values

  # Structure data
  X_data_columns = ['spread', 'ticker_1_rets', 'ticker_2_rets']
  df = df[X_data_columns]
  X_data = df.iloc[:, :]

  # Load ML Model
  xbg_classifier = XGBClassifier()
  xbg_classifier.load_model(f"models/model.json")
  preds = xbg_classifier.predict(X_data)
  preds_proba = xbg_classifier.predict_proba(X_data)

  # Add to dataframe
  df["PRED"] = preds
  df[["PROBA_0", "PROBA_1"]] = preds_proba
  df["ACTUAL"] = actuals

  # Calculate overall accuracy
  df.loc[df["PRED"] == df["ACTUAL"], "CORRECT"] = 1
  df.loc[df["PRED"] != df["ACTUAL"], "CORRECT"] = 0
  acc_perc = df["CORRECT"].sum() / len(df)
  print(f"Accuracy: {round(acc_perc * 100, 1)}%")
