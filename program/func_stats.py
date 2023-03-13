from constants import WINDOW, TICKER_1, TICKER_2
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm
import pandas as pd
import numpy as np
import math


# Calculate Half Life
# https://www.pythonforfinance.net/2016/05/09/python-backtesting-mean-reversion-part-2/
def calculate_half_life(spread):
  df_spread = pd.DataFrame(spread, columns=["spread"])
  spread_lag = df_spread.spread.shift(1)
  spread_lag.iloc[0] = spread_lag.iloc[1]
  spread_ret = df_spread.spread - spread_lag
  spread_ret.iloc[0] = spread_ret.iloc[1]
  spread_lag2 = sm.add_constant(spread_lag)
  model = sm.OLS(spread_ret, spread_lag2)
  res = model.fit()
  halflife = round(-np.log(2) / res.params[1], 0)
  return halflife


# Calculate ZScore
def calculate_zscore(spread):
  spread_series = pd.Series(spread)
  mean = spread_series.rolling(center=False, window=WINDOW).mean()
  std = spread_series.rolling(center=False, window=WINDOW).std()
  x = spread_series.rolling(center=False, window=1).mean()
  zscore = (x - mean) / std
  return zscore


# Calculate Cointegration
def calculate_cointegration_metrics(series_1, series_2):
  series_1 = np.array(series_1).astype(np.float64)
  series_2 = np.array(series_2).astype(np.float64)
  coint_flag = 0
  coint_res = coint(series_1, series_2)
  coint_t = coint_res[0]
  p_value = coint_res[1]
  critical_value = coint_res[2][1]
  model = sm.OLS(series_1, series_2).fit()
  hedge_ratio = model.params[0]
  spread = series_1 - (hedge_ratio * series_2)
  t_check = coint_t < critical_value
  z_score = calculate_zscore(spread)
  half_life = calculate_half_life(spread)
  coint_flag = 1 if p_value < 0.05 and t_check else 0
  return p_value, t_check, z_score.values[-1], spread[-1]


# Save data with rolling metrics
def store_rolling_metrics():

  # Read Data
  df = pd.read_csv("data/combined.csv")
  df.dropna(inplace=True)
  df.set_index("datetime", inplace=True)

  # Initialize periods and lists
  # Assumes hourly data on 14 days of data
  periods = 24 * 14
  p_values = [math.nan] * periods
  t_checks = [math.nan] * periods
  z_scores = [math.nan] * periods
  spreads = [math.nan] * periods
  corrs = [math.nan] * periods

  # Get Metrics
  try:
    for i in range(len(df) - periods):

      # Construct Series
      series_1 = df[f"{TICKER_1}_close"].astype(float)[i:i+periods]
      series_2 = df[f"{TICKER_2}_close"].astype(float)[i:i+periods]

      # Get Cointegration Metrics
      p_value, t_check, z_score, spread = calculate_cointegration_metrics(series_1, series_2)
      p_values.append(p_value)
      t_checks.append(t_check)
      z_scores.append(z_score)
      spreads.append(spread)

      # Get correlation
      corr = np.corrcoef(series_1, series_2)[0, 1]
      corrs.append(corr)

    # Create Metrics Dataframe
    df["coint_p_value"] = p_values
    df["coint_t_check"] = t_checks
    df["z_score"] = z_scores
    df["spread"] = spreads
    df["corr"] = corrs

    # Get Close Returns
    df["ticker_1_rets"] = df[f"{TICKER_1}_close"].pct_change()
    df["ticker_2_rets"] = df[f"{TICKER_2}_close"].pct_change()

    # Get Spread Returns
    scaler = MinMaxScaler()
    df["scaler"] = scaler.fit_transform(df[["spread"]])
    df["spread_rets"] = df["scaler"].pct_change()

    # Get Range
    df[f"{TICKER_1}_range"] = df[f"{TICKER_1}_high"] - df[f"{TICKER_1}_low"]
    df[f"{TICKER_2}_range"] = df[f"{TICKER_2}_high"] - df[f"{TICKER_2}_low"]

    # Get Volatility
    df[f"{TICKER_1}_volatility"] = df[f"{TICKER_1}_close"].rolling(window=WINDOW).std()
    df[f"{TICKER_2}_volatility"] = df[f"{TICKER_2}_close"].rolling(window=WINDOW).std()
    df[f"spread_volatility"] = df["scaler"].rolling(window=WINDOW).std()

    # Remove columns and na
    df.drop(columns=[f"{TICKER_1}_high", f"{TICKER_1}_low", f"{TICKER_2}_high", f"{TICKER_2}_low"], inplace=True)
    df.dropna(inplace=True)

    # Save df
    df.to_csv("data/combined_metrics.csv")
    print(df)
    del(df)

  except Exception as e:
    print("Error: ", e)
    exit(1)



