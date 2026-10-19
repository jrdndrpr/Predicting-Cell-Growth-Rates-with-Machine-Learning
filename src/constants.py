# =======================================================================
# CONSTANTS
# Define global constants in this file.
# =======================================================================

import os

PH_LIMITS = (6.8, 7.2)
TEMPERATURE_LIMITS = (35.0, 37.0)

DATASET_DIR = "datasets"
DATASET_FILENAME = "dataset_cell_growth_4factor.csv"
TABLES_DIR = "tables"
FIGURES_DIR = "figures"
RANDOM_SEED = 1

PARAM_GRID_POLY = {
    "polynomialfeatures__degree": [1, 2, 3, 4]
}

PARAM_GRID_KNN = {
    "kneighborsregressor__n_neighbors": [3, 5, 7, 9, 11, 15, 21],
    "kneighborsregressor__weights": ["uniform", "distance"]
}