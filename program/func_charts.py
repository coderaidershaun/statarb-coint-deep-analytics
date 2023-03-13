import matplotlib.pyplot as plt
import pandas as pd


# Plot histogramn
def plot_histogram(series, title):
  plt.hist(series, bins=50)
  plt.xlabel(f"Series")
  plt.ylabel("Frequency")
  plt.savefig(f"images/{title}.png")
  plt.clf()


# Plot line chart
def plot_multiaxis_line_chart(series1, title1, series2, title2):

  # Get x
  x = list(range(10))

  # Create graph with first axis
  fig, ax1 = plt.subplots()

  # Axis 1
  ax1.plot(series1, 'b-')
  ax1.set_xlabel('X data')
  ax1.set_ylabel(title1, color='b')
  ax1.tick_params('y', colors='b')

  # Create a second axis
  ax2 = ax1.twinx()
  ax2.plot(series2, 'g-')
  ax2.set_ylabel(title2, color='g')
  ax2.tick_params('y', colors='g')

  # Sace plot
  plt.savefig(f"images/line-chart-{title1}-{title2}.png")
  plt.clf()
