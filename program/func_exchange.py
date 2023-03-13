from dydx3.constants import API_HOST_MAINNET
from dydx3 import Client

from constants import RESOLUTION, TICKER_1, TICKER_2
from func_utils import get_ISO_times
from decouple import config
from web3 import Web3
import pandas as pd
import time

from pprint import pprint


# Connect to DYDX
def connect_dydx():
  HTTP_PROVIDER = config("HTTP_PROVIDER")
  client = Client(
      host=API_HOST_MAINNET,
      web3=Web3(Web3.HTTPProvider(HTTP_PROVIDER))
  )
  return client


# Get DYDX OHLC Data
def get_dydx_historical_ohlc_data(ticker, ISO_TIMES):
    client = connect_dydx()

    # Extract historical price data for each timeframe
    close_prices = []
    counts = 0
    for timeframe in ISO_TIMES.keys():

        # Protect rate limits and notify operator
        counts += 1
        time.sleep(1)
        print(f"Fetching {ticker} for count: {counts}")

        # Confirm times needed
        tf_obj = ISO_TIMES[timeframe]
        from_iso = tf_obj["from_iso"]
        to_iso = tf_obj["to_iso"]

        # Get data
        candles = client.public.get_candles(
            market=ticker,
            resolution=RESOLUTION,
            from_iso=from_iso,
            to_iso=to_iso,
            limit=100
        )

        # Structure data
        if "candles" in candles.data:
            if len(candles.data["candles"]) > 0:
                for candle in candles.data["candles"]:
                    close_prices.append(
                        {
                            "datetime": candle["startedAt"], 
                            f"{ticker}_open": float(candle["open"]),
                            f"{ticker}_high": float(candle["high"]),
                            f"{ticker}_low": float(candle["low"]),
                            f"{ticker}_close": float(candle["close"])
                        }
                    )

    # Reverse data array
    if len(close_prices) > 0:
        close_prices.reverse()

    # Return prices
    return close_prices


# Handle calling correct data
def store_data():

    # Get times for data history
    ISO_TIMES = get_ISO_times()
    
    # Get Data
    prices_1 = []
    prices_2 = []

    # Get first ticker price history
    prices_1 = get_dydx_historical_ohlc_data(TICKER_1, ISO_TIMES)

    # Store price1 history
    if len(prices_1) > 0:
        df_1 = pd.DataFrame(prices_1)
        df_1.set_index("datetime", inplace=True)
        df_1.to_csv("data/ticker_1.csv")

    # Get second ticker price history
    prices_2 = get_dydx_historical_ohlc_data(TICKER_2, ISO_TIMES)

    # Store price2 history
    if len(prices_2) > 0:
        df_2 = pd.DataFrame(prices_2)
        df_2.set_index("datetime", inplace=True)
        df_2.to_csv("data/ticker_2.csv")

    # Merge Data, Save Data, Clear from  Memory
    if len(df_1) == len(df_2):
        df_merge = pd.merge(df_1, df_2, left_index=True, right_index=True)
        df_merge.to_csv("data/combined.csv")
        print(df_merge)
        del(df_1)
        del(df_2)
        del(df_merge)
    else:
        print("Error: Ticker data not same length")
        exit(1)


# Get current data
def get_current_dydx_data():
    ISO_TIMES = get_ISO_times(is_predictions=True)

    # Get dataset - ticker 1
    prices_1 = get_dydx_historical_ohlc_data(TICKER_1, ISO_TIMES)
    df_1 = pd.DataFrame(prices_1)
    df_1.set_index("datetime", inplace=True)

    # Get dataset - ticker 2
    prices_2 = get_dydx_historical_ohlc_data(TICKER_2, ISO_TIMES)
    df_2 = pd.DataFrame(prices_2)
    df_2.set_index("datetime", inplace=True)

    # Merge dataframes
    if len(df_1) == len(df_2) and len(df_1) > 0:
        df_merge= pd.merge(df_1, df_2, left_index=True, right_index=True)
        del(df_1)
        del(df_2)
        return df_merge
    
    # Handle Length Error
    else:
        print("Error: Data not fully retrieved and matching in length")
        exit(1)
