# =======================================================================
# MAIN PROGRAM
# Your code starts here. This is the file that will be run.
# =======================================================================

import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from src.classes import Batch, CellGrowthPredictor
from src.constants import *
import src.functions as sf


def main():
  ph_optimal = sf.calculate_average(PH_LIMITS)
  temperature_optimal = sf.calculate_average(TEMPERATURE_LIMITS)

  batch = Batch(batch_id=1, ph_opt=ph_optimal, temperature_opt=temperature_optimal)
  batch.display_info()

  # Locate dataset
  dataset_path = os.path.join(DATASET_DIR, DATASET_FILENAME)
  if not os.path.exists(dataset_path) and os.path.exists(DATASET_FILENAME):
    dataset_path = DATASET_FILENAME

  # Load dataset
  X, y, feature_names, target_name = sf.load_cell_growth_data(dataset_path)

  # Train-test split
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=RANDOM_SEED
  )

  polynomial_records = []

  # 1. Standard Linear Regression (Baseline)
  linear_pipe = make_pipeline(StandardScaler(), LinearRegression())
  linear_predictor = CellGrowthPredictor(linear_pipe, name="Linear Regression")
  lin_metrics = sf.evaluate_model(
      linear_predictor, X_train, y_train, X_test, y_test
  )
  linear_record = {
      "Model": "Standard Linear Regression",
      "Hyperparameters": "None",
      "Train RMSE [d^-1]": lin_metrics["train_rmse"],
      "Test RMSE [d^-1]": lin_metrics["test_rmse"],
      "Train R^2": lin_metrics["train_r2"],
      "Test R^2": lin_metrics["test_r2"],
  }
  polynomial_records.append(linear_record)

  # 2. Individual Polynomial Degrees (1 to 4)
  for deg in [1, 2, 3, 4]:
    poly_deg_pipe = make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=deg, include_bias=False),
        LinearRegression(),
    )
    poly_deg_predictor = CellGrowthPredictor(
        poly_deg_pipe, name=f"Polynomial Regression (Deg {deg})"
    )
    deg_metrics = sf.evaluate_model(
        poly_deg_predictor, X_train, y_train, X_test, y_test
    )
    polynomial_records.append({
        "Model": f"Polynomial Regression (Deg {deg})",
        "Hyperparameters": f"Degree={deg}",
        "Train RMSE [d^-1]": deg_metrics["train_rmse"],
        "Test RMSE [d^-1]": deg_metrics["test_rmse"],
        "Train R^2": deg_metrics["train_r2"],
        "Test R^2": deg_metrics["test_r2"],
    })

  # Export polynomial-only performance table
  sf.export_table(
      polynomial_records,
      tables_dir=TABLES_DIR,
      filename="table_polynomial_performance.csv",
  )

  # 3. kNN Regression with GridSearchCV & Sweep
  knn_pipe = make_pipeline(StandardScaler(), KNeighborsRegressor())
  gscv_knn = GridSearchCV(
      knn_pipe, PARAM_GRID_KNN, cv=5, scoring="neg_root_mean_squared_error"
  )
  gscv_knn.fit(X_train, y_train)
  knn_predictor = CellGrowthPredictor(
      gscv_knn.best_estimator_, name="kNN Regression"
  )
  knn_metrics = sf.evaluate_model(
      knn_predictor, X_train, y_train, X_test, y_test
  )
  best_k = gscv_knn.best_params_["kneighborsregressor__n_neighbors"]
  best_weight = gscv_knn.best_params_["kneighborsregressor__weights"]

  knn_sweep_records = sf.evaluate_knn_sweep(
      X_train, y_train, X_test, y_test, k_values=[3, 5, 7, 9, 11, 15, 20]
  )
  sf.export_table(
      knn_sweep_records, tables_dir=TABLES_DIR, filename="table_knn_sweep.csv"
  )

  # 4. Best Model Comparison Table (Using Degree 2 as Optimal Polynomial Model)
  best_poly_record = polynomial_records[2]  # Degree 2 Polynomial
  best_knn_record = {
      "Model": "Optimized kNN Regression",
      "Hyperparameters": f"k={best_k}, weights={best_weight}",
      "Train RMSE [d^-1]": knn_metrics["train_rmse"],
      "Test RMSE [d^-1]": knn_metrics["test_rmse"],
      "Train R^2": knn_metrics["train_r2"],
      "Test R^2": knn_metrics["test_r2"],
  }
  best_comparison_records = [linear_record, best_poly_record, best_knn_record]
  sf.export_table(
      best_comparison_records,
      tables_dir=TABLES_DIR,
      filename="table_best_model_comparison.csv",
  )

  # Generate separate model comparison bar charts
  sf.plot_model_comparison_rmse(best_comparison_records, FIGURES_DIR)
  sf.plot_model_comparison_r2(best_comparison_records, FIGURES_DIR)

  # Generate figures silently using optimal Degree 2 Polynomial model
  poly_best_pipe = make_pipeline(
      StandardScaler(),
      PolynomialFeatures(degree=2, include_bias=False),
      LinearRegression(),
  )
  poly_best_predictor = CellGrowthPredictor(
      poly_best_pipe, name="Polynomial Regression"
  )
  poly_best_predictor.fit(X_train, y_train)

  sf.plot_actual_vs_predicted(
      y_test,
      poly_best_predictor.predict(X_test),
      "Polynomial Regression",
      FIGURES_DIR,
  )
  sf.plot_actual_vs_predicted(
      y_test, knn_predictor.predict(X_test), "kNN Regression", FIGURES_DIR
  )
  sf.plot_feature_correlations(dataset_path, FIGURES_DIR)
  sf.plot_polynomial_complexity(
      X_train, y_train, X_test, y_test, FIGURES_DIR
  )
  sf.plot_knn_complexity(
      X_train, y_train, X_test, y_test, figures_dir=FIGURES_DIR
  )
  sf.plot_process_variable_effects(
      poly_best_predictor, X_train, feature_names, FIGURES_DIR
  )


if __name__ == "__main__":
  main()