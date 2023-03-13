"""
  CONSTANTS
  Contains settings and configurations
"""

# ACTIVATIONS
IS_STORE_DATA = False
IS_STORE_ROLLING_METRICS = False
IS_ANALYSE_METRICS = False
IS_MACHINE_LEARNING = True
IS_PREDICTIONS = False

# FIXED ITEMS
# Changes to these would require changes to code
RESOLUTION = "1HOUR"

# VARIABLE ITEMS
# Change these to change how the code behaves
TICKER_1 = "EOS-USD" # Asset 1
TICKER_2 = "LINK-USD" # Asset 2
PERIOD = "1YEAR" # 1YEAR, 6MONTH - represents amount of data to sort through
WINDOW = 21 # Represents the rolling amount of periods to take into account
OOS_SIZE = 500 # Represents rows to ignore on the end of the dataset for training
