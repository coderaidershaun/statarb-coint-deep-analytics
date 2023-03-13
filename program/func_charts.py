import matplotlib.pyplot as plt
from func_utils import clean_infs
import pandas as pd


# Plot histogramn
def plot_histogram(series, title):
  plot_list = clean_infs(series)
  plt.hist(plot_list, bins=50)
  plt.xlabel(f"Series")
  plt.ylabel("Frequency")
  plt.savefig(f"images/{title}.png")


# Plot line chart
def plot_multiaxis_line_chart(series1, title1, series2, title2):
  pass
