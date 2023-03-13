from datetime import datetime, timedelta
from constants import PERIOD, WINDOW


# Format time
def format_time(timestamp):
  return timestamp.replace(microsecond=0).isoformat()


# Get ISO Times
def get_ISO_times(is_predictions=False):

  # Confirm count of exchange data pulls needed
  # This is because DYDX limits each data pull to x100 candles
  periods = 0
  if not is_predictions:
    if PERIOD == "1YEAR":
      periods = int((365 * 24) / 100) + 1
    elif PERIOD == "6MONTH":
      periods = int((183 * 24) / 100) + 1
    else:
      # Defaults to 1 year
      periods = int((365 * 24) / 100) + 1
  
  # Get just the last 300 data points for making predictions
  else:
    periods = 1

  # Initialize timestamp
  date_end = datetime.now()

  # Construct dictionary of times
  times_dict = {}
  for i in range(periods):
    date_start_delta = date_end - timedelta(hours=100)
    times_dict[f"range_{i}"] = {
      "from_iso": format_time(date_start_delta),
      "to_iso": format_time(date_end)
    }
    date_end = date_start_delta
    
  # Return result
  return times_dict


# Clean list from inf values
def clean_infs(arr):
  for i in range(1, len(arr)):
    if arr[i] == float('inf'):
        arr[i] = arr[i-1]
  return arr
