# =======================================================================
# CLASSES
# Define custom classes in this file.
# =======================================================================

class Batch:
    def __init__(self, batch_id, ph_opt, temperature_opt):
        self.batch_id = batch_id
        self.ph_opt = ph_opt
        self.temperature_opt = temperature_opt

    def display_info(self):
        print(f"Batch ID: {self.batch_id:4d} | "
              f"Optimal pH: {self.ph_opt:4.1f} | "
              f"Optimal Temperature: {self.temperature_opt:4.1f} C")


class CellGrowthPredictor:
    """Encapsulates the machine learning regression pipeline for cell growth prediction."""
    def __init__(self, model, name="CellGrowthPredictor"):
        self.model = model
        self.name = name
        self.is_fitted = False

    def fit(self, X, y):
        """Fits the underlying model pipeline to training data."""
        self.model.fit(X, y)
        self.is_fitted = True

    def predict(self, X):
        """Generates predictions using the fitted model pipeline."""
        if not self.is_fitted:
            raise ValueError(f"Model {self.name} must be fitted before calling predict().")
        return self.model.predict(X)