from constants import IS_STORE_DATA, IS_STORE_ROLLING_METRICS, IS_ANALYSE_METRICS
from func_exchange import store_data
from func_stats import store_rolling_metrics
from func_charts import plot_histogram, plot_line_chart
import pandas as pd

# ENTRYPOINT
if __name__ == "__main__":

    # Get OHLC Price Data
    if IS_STORE_DATA:
        print("Gettign data, please allow a few short minutes...")
        store_data()

    # Get Rolling Metrics
    if IS_STORE_ROLLING_METRICS:
        print("Gettign metrics, please allow a few short minutes...")
        store_rolling_metrics()

    # Read In Metrics Data
    df = pd.read_csv("data/combined_metrics.csv")
    df.dropna(inplace=True)
    df.set_index("datetime", inplace=True)

    # Plot charts
    if IS_ANALYSE_METRICS:
        spreads = df["spread"].values
        plot_histogram(spreads, "spread-hist")
        plot_line_chart(df)

    # Get HMM States (volatility and returns)
    # if HMM_STATES:

    # Get HMM Switching Data

    # Construct Kalman Filters

    # 
