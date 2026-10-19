# =======================================================================
# FUNCTIONS
# Define reusable helper functions in this file.
# =======================================================================

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def calculate_average(values):
  return sum(values) / len(values)


def load_cell_growth_data(filepath):
  """Loads dataset and extracts feature matrix and target vector."""
  if not os.path.exists(filepath):
    raise FileNotFoundError(f"Dataset not found at {filepath}")
  df = pd.read_csv(filepath)
  target_col = df.columns[0]
  feature_cols = df.columns[1:]

  X = df.loc[:, feature_cols].to_numpy()
  y = df.loc[:, target_col].to_numpy()
  return X, y, feature_cols, target_col


def evaluate_model(predictor, X_train, y_train, X_test, y_test):
  """Trains a CellGrowthPredictor and computes train/test RMSE and R^2."""
  predictor.fit(X_train, y_train)
  y_train_pred = predictor.predict(X_train)
  y_test_pred = predictor.predict(X_test)

  metrics = {
      "train_rmse": root_mean_squared_error(y_train, y_train_pred),
      "test_rmse": root_mean_squared_error(y_test, y_test_pred),
      "train_r2": r2_score(y_train, y_train_pred),
      "test_r2": r2_score(y_test, y_test_pred),
  }
  return metrics


def export_table(
    results_list,
    tables_dir="tables",
    filename="table_model_performance.csv",
):
  """Exports any results list to a CSV file in the tables directory silently."""
  os.makedirs(tables_dir, exist_ok=True)
  df_export = pd.DataFrame(results_list)
  filepath = os.path.join(tables_dir, filename)
  df_export.to_csv(filepath, index=False)
  return df_export


def evaluate_knn_sweep(
    X_train, y_train, X_test, y_test, k_values=[3, 5, 7, 9, 11, 15, 21]
):
  """Evaluates multiple k values and weights for kNN regression."""
  records = []
  for k in k_values:
    for weights in ["uniform", "distance"]:
      pipe = make_pipeline(
          StandardScaler(), KNeighborsRegressor(n_neighbors=k, weights=weights)
      )
      predictor = type(
          "Dummy",
          (object,),
          {"fit": pipe.fit, "predict": pipe.predict},
      )()
      metrics = evaluate_model(predictor, X_train, y_train, X_test, y_test)
      records.append({
          "Model": "kNN Regression",
          "Neighbors (k)": k,
          "Weighting": weights,
          "Train RMSE [d^-1]": metrics["train_rmse"],
          "Test RMSE [d^-1]": metrics["test_rmse"],
          "Train R^2": metrics["train_r2"],
          "Test R^2": metrics["test_r2"],
      })
  return records


def plot_actual_vs_predicted(y_true, y_pred, model_name, figures_dir="figures"):
  """Generates and saves model prediction parity plots."""
  os.makedirs(figures_dir, exist_ok=True)
  fig, ax = plt.subplots(figsize=(6.5, 5), dpi=200, layout="constrained")

  ax.scatter(
      y_true,
      y_pred,
      color="tab:blue",
      edgecolors="black",
      alpha=0.7,
      s=32,
      label="Predictions",
  )
  min_val = min(min(y_true), min(y_pred))
  max_val = max(max(y_true), max(y_pred))
  ax.plot(
      [min_val, max_val],
      [min_val, max_val],
      color="black",
      linestyle="--",
      linewidth=1.5,
      label="Ideal (y = y_pred)",
  )

  ax.set_xlabel("Actual Specific Growth Rate [d^-1]", fontsize=12)
  ax.set_ylabel("Predicted Specific Growth Rate [d^-1]", fontsize=12)
  ax.set_title(f"Actual vs Predicted: {model_name}", fontsize=14)
  ax.legend(fontsize=10)
  ax.tick_params(axis="both", which="major", labelsize=10)

  filename_fig = (
      f"figure_actual_vs_predicted_{model_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('=', '_')}.png"
  )
  fig.savefig(os.path.join(figures_dir, filename_fig))
  plt.close(fig)


def plot_feature_correlations(filepath, figures_dir="figures"):
  """Generates and saves a 2x2 4-panel scatter plot of features vs specific growth rate."""
  os.makedirs(figures_dir, exist_ok=True)
  df = pd.read_csv(filepath)
  target = df.columns[0]
  features = df.columns[1:]

  fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=200, layout="constrained")
  axes_flat = axes.flatten()

  for i, col in enumerate(features):
    axes_flat[i].scatter(
        df[col],
        df[target],
        alpha=0.6,
        s=24,
        edgecolors="black",
        linewidth=0.5,
        color="tab:blue",
    )
    axes_flat[i].set_xlabel(col, fontsize=12)
    axes_flat[i].set_ylabel(target, fontsize=12)
    axes_flat[i].tick_params(axis="both", which="major", labelsize=10)

  filepath_fig = os.path.join(figures_dir, "figure_feature_correlations.png")
  fig.savefig(filepath_fig)
  plt.close(fig)


def plot_model_comparison_rmse(results_list, figures_dir="figures"):
  """Generates and saves a bar chart comparing Test RMSE across compared models (without data labels)."""
  os.makedirs(figures_dir, exist_ok=True)
  df = pd.DataFrame(results_list)

  models = df["Model"].tolist()
  test_rmses = df["Test RMSE [d^-1]"].tolist()

  fig, ax = plt.subplots(figsize=(6.5, 5), dpi=200, layout="constrained")
  color = "tab:blue"
  ax.bar(
      models, test_rmses, color=color, alpha=0.8, edgecolor="black", width=0.5
  )
  ax.set_xlabel("Model", fontsize=12)
  ax.set_ylabel("Test RMSE [d^-1]", fontsize=12)
  ax.set_title("Model Comparison: Test RMSE", fontsize=14)

  ax.set_xticks(range(len(models)))
  ax.set_xticklabels(models, rotation=15, ha="right", fontsize=10)
  ax.tick_params(axis="both", which="major", labelsize=10)

  filepath = os.path.join(figures_dir, "figure_model_comparison_rmse.png")
  fig.savefig(filepath)
  plt.close(fig)


def plot_model_comparison_r2(results_list, figures_dir="figures"):
  """Generates and saves a bar chart comparing Test R^2 across compared models (without data labels)."""
  os.makedirs(figures_dir, exist_ok=True)
  df = pd.DataFrame(results_list)

  models = df["Model"].tolist()
  test_r2s = df["Test R^2"].tolist()

  fig, ax = plt.subplots(figsize=(6.5, 5), dpi=200, layout="constrained")
  color = "tab:orange"
  ax.bar(
      models, test_r2s, color=color, alpha=0.8, edgecolor="black", width=0.5
  )
  ax.set_xlabel("Model", fontsize=12)
  ax.set_ylabel("Test R^2", fontsize=12)
  ax.set_title("Model Comparison: Test R^2", fontsize=14)

  ax.set_xticks(range(len(models)))
  ax.set_xticklabels(models, rotation=15, ha="right", fontsize=10)
  ax.tick_params(axis="both", which="major", labelsize=10)
  ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

  filepath = os.path.join(figures_dir, "figure_model_comparison_r2.png")
  fig.savefig(filepath)
  plt.close(fig)


def plot_polynomial_complexity(
    X_train, y_train, X_test, y_test, figures_dir="figures"
):
  """Generates and saves a training vs test performance curve across polynomial degrees 1 to 4."""
  os.makedirs(figures_dir, exist_ok=True)
  degrees = [1, 2, 3, 4]
  train_rmses, test_rmses = [], []

  for deg in degrees:
    pipe = make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=deg, include_bias=False),
        LinearRegression(),
    )
    pipe.fit(X_train, y_train)

    train_rmses.append(
        root_mean_squared_error(y_train, pipe.predict(X_train))
    )
    test_rmses.append(root_mean_squared_error(y_test, pipe.predict(X_test)))

  fig, ax = plt.subplots(figsize=(7, 5), dpi=200, layout="constrained")
  ax.plot(
      degrees,
      train_rmses,
      marker="o",
      linestyle="-",
      linewidth=2,
      label="Training RMSE",
  )
  ax.plot(
      degrees,
      test_rmses,
      marker="s",
      linestyle="--",
      linewidth=2,
      label="Testing RMSE",
  )

  ax.set_xlabel("Polynomial Degree", fontsize=12)
  ax.set_ylabel("Root Mean Squared Error [d^-1]", fontsize=12)
  ax.set_title(
      "Polynomial Regression: Model Complexity & Generalization", fontsize=13
  )
  ax.set_xticks(degrees)
  ax.legend(fontsize=10)
  ax.tick_params(axis="both", which="major", labelsize=10)

  filepath = os.path.join(figures_dir, "figure_polynomial_complexity.png")
  fig.savefig(filepath)
  plt.close(fig)


def plot_knn_complexity(
    X_train,
    y_train,
    X_test,
    y_test,
    k_values=[3, 5, 7, 9, 11, 15, 21],
    figures_dir="figures",
):
  """Generates and saves a test RMSE curve across k values for kNN regression with uniform and distance weights."""
  os.makedirs(figures_dir, exist_ok=True)
  uniform_rmses, distance_rmses = [], []

  for k in k_values:
    pipe_u = make_pipeline(
        StandardScaler(), KNeighborsRegressor(n_neighbors=k, weights="uniform")
    )
    pipe_u.fit(X_train, y_train)
    uniform_rmses.append(
        root_mean_squared_error(y_test, pipe_u.predict(X_test))
    )

    pipe_d = make_pipeline(
        StandardScaler(), KNeighborsRegressor(n_neighbors=k, weights="distance")
    )
    pipe_d.fit(X_train, y_train)
    distance_rmses.append(
        root_mean_squared_error(y_test, pipe_d.predict(X_test))
    )

  fig, ax = plt.subplots(figsize=(7, 5), dpi=200, layout="constrained")
  ax.plot(
      k_values,
      uniform_rmses,
      marker="o",
      linestyle="-",
      linewidth=2,
      label="Uniform Weights",
  )
  ax.plot(
      k_values,
      distance_rmses,
      marker="s",
      linestyle="--",
      linewidth=2,
      label="Distance Weights",
  )

  ax.set_xlabel("Number of Neighbors (k)", fontsize=12)
  ax.set_ylabel("Testing Root Mean Squared Error [d^-1]", fontsize=12)
  ax.set_title(
      "kNN Regression: Hyperparameter Sweep (k vs. Test RMSE)", fontsize=13
  )
  ax.set_xticks(k_values)
  ax.legend(fontsize=10)
  ax.tick_params(axis="both", which="major", labelsize=10)

  filepath = os.path.join(figures_dir, "figure_knn_complexity.png")
  fig.savefig(filepath)
  plt.close(fig)


def plot_process_variable_effects(
    predictor, X_train, feature_names, figures_dir="figures"
):
  """Generates a 2x2 subplot showing the marginal effect of each feature on predicted growth rate."""
  os.makedirs(figures_dir, exist_ok=True)

  mean_vector = X_train.mean(axis=0)
  fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=200, layout="constrained")
  axes_flat = axes.flatten()

  for i, col_name in enumerate(feature_names):
    ax = axes_flat[i]
    feat_min, feat_max = X_train[:, i].min(), X_train[:, i].max()
    sweep_vals = np.linspace(feat_min, feat_max, 100)

    X_sweep = np.tile(mean_vector, (100, 1))
    X_sweep[:, i] = sweep_vals

    y_pred = predictor.predict(X_sweep)

    ax.plot(sweep_vals, y_pred, linewidth=2, color="tab:blue")
    ax.set_title(f"Effect of {col_name}", fontsize=12)
    ax.set_xlabel(f"{col_name}", fontsize=11)
    ax.set_ylabel("Predicted Growth Rate [d^-1]", fontsize=11)
    ax.grid(True, linestyle="-", alpha=0.3)
    ax.tick_params(axis="both", which="major", labelsize=10)

  fig.suptitle(
      "Effect of Process Variables on Predicted Growth Rate", fontsize=14, y=1.03
  )
  filepath = os.path.join(figures_dir, "figure_process_variable_effects.png")
  fig.savefig(filepath, bbox_inches="tight")
  plt.close(fig)