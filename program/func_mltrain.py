from constants import TICKER_1, TICKER_2
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold
from sklearn.metrics import classification_report
from xgboost import XGBClassifier, plot_tree
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from pprint import pprint


# Run ML Predictions
def train_model(df):

  # Define columns
  # 'z_score', 'spread_volatility', f"{TICKER_1}_volatility", f"{TICKER_2}_volatility", 'coint_p_value', , f"{TICKER_1}_range", f"{TICKER_2}_range", 'corr', 'spread_rets'
  X_data_columns = ['spread', 'ticker_1_rets', 'ticker_2_rets']

  # Keep relevant columns and ignore last 300 rows of data
  # We will get the most recent 300 rows later on to ensure the model stands true
  df = df[X_data_columns].iloc[:-300,:]

  # Add Targets
  df.loc[df["spread"].shift(-1) > df["spread"], "TARGET"] = 1
  df.loc[df["spread"].shift(-1) <= df["spread"], "TARGET"] = 0

  # Drop NA
  df.dropna(inplace=True)
  df.replace([np.inf, -np.inf], np.nan, inplace=True)

  # Define X_Data
  X_data = df.iloc[:, :-1]
  y_data = df.iloc[:, -1]

  # Initialize params
  ne = 30
  md = 3
  lr = 0.1
  gm = 0.03
  test_size_rate = 0.8

  # Train Test Split
  X_train, X_test, y_train, y_test = train_test_split(
      X_data,
      y_data,
      random_state=0,
      test_size=test_size_rate,
      shuffle=True)
  
  # For binary classification
  objective = "binary:logistic"
  eval_metric = "logloss"
  eval_metric_list = ["error", "logloss"]

  # Evaluation
  eval_metric = "aucpr"
  eval_metric_list.append(eval_metric)
  scoring = 'precision'

  # Build Classification Model with Initial Hyperparams
  classifier = XGBClassifier(
    objective=objective,
    booster="gbtree",
    n_estimators=ne,
    learning_rate=lr,
    max_depth=md,
    subsample=0.8,
    colsample_bytree=1,
    gamma=gm,
    random_state=1,
  )

  # Fit Model
  eval_set = [(X_train, y_train), (X_test, y_test)]
  classifier.fit(
    X_train,
    y_train,
    eval_metric=eval_metric_list,
    eval_set=eval_set,
    verbose=False,
  )

  # Extract predictions
  train_yhat_preds = classifier.predict(X_train)
  test_yhat_preds = classifier.predict(X_test)
  train_yhat_proba = classifier.predict_proba(X_train)
  test_yhat_proba = classifier.predict_proba(X_test)

  # Set K-Fold Cross Validation levels
  cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)

  # Training Results
  train_cross_val_score = cross_val_score(classifier, X_train, y_train, scoring=scoring, cv=cv, n_jobs=-1)
  train_summary_report = classification_report(y_train, train_yhat_preds, output_dict=True, zero_division=False)
  test_summary_report = classification_report(y_test, test_yhat_preds, output_dict=True, zero_division=False)

  # Standard deviation
  std_dev_perc = train_cross_val_score.std() * 100
  avg_score_perc = train_cross_val_score.mean() * 100

  # Show key metrics
  print("")
  print("Std Deviation %: ", round(std_dev_perc, 5))
  print("Avg Accuracy %: ", round(avg_score_perc, 5))
  print("Train Precision %: ", round(train_summary_report["1.0"]["precision"], 5) * 100)
  print("Test Precision %: ", round(test_summary_report["1.0"]["precision"], 5) * 100)
  print("")

  # Print Summary
  pprint(test_summary_report)

  # Save model
  classifier.save_model("models/model.json")
  pprint("Model saved")

  # # Feature importance
  # importance_features = classifier.feature_importances_
  # plt.title('Feature Importance')
  # plt.bar(df.columns[:-1], importance_features)
  # plt.show()

  # # Plot tree
  # plot_tree(classifier, num_trees=0)
  # plt.show()

  # # Other useful plots
  # training_results = classifier.evals_result()
  # validation_0_error = training_results['validation_0'][eval_metric_list[0]]
  # validation_1_error = training_results['validation_1'][eval_metric_list[0]]
  # validation_0_logloss = training_results['validation_0'][eval_metric_list[1]]
  # validation_1_logloss = training_results['validation_1'][eval_metric_list[1]]
  # validation_0_auc = training_results['validation_0'][eval_metric_list[2]]
  # validation_1_auc = training_results['validation_1'][eval_metric_list[2]]

  # # Plots
  # plt.title('Error')
  # plt.plot(validation_0_error)
  # plt.plot(validation_1_error)
  # plt.show()
