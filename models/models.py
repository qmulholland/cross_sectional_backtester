import numpy as np


class LinearSignalModel:
    def __init__(self, weights: np.ndarray):
        self.weights = weights

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.weights
