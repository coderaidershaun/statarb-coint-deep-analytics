from constants import IS_STORE_DATA, IS_STORE_ROLLING_METRICS, IS_ANALYSE_METRICS, IS_MACHINE_LEARNING, IS_PREDICTIONS
from func_exchange import store_data, get_dydx_historical_ohlc_data
from func_stats import store_rolling_metrics
from func_charts import plot_histogram, plot_multiaxis_line_chart
from func_mltrain import train_model
from func_mlpredict import make_predictions
from func_utils import clean_infs
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
        spreads = clean_infs(df["spread"].values)
        plot_histogram(spreads, "spread-hist")

        pvals = clean_infs(df["coint_p_value"].values)
        plot_multiaxis_line_chart(spreads, "spread", pvals, "pvalue")

    # Train Machine Learning Model
    if IS_MACHINE_LEARNING:
        train_model(df)

    # Get Predictions for recent data
    if IS_PREDICTIONS:
        make_predictions()
