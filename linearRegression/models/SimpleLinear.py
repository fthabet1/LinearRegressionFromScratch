import pandas as pd
import numpy as np
from linearRegression.models.BaseModel import BaseModel
from linearRegression.optimizers.GradientDescent import GradientDescent

class SimpleLinearModel(BaseModel):

    def __init__(self):
        super().__init__()
        self.optimizer = GradientDescent()

    def fit(self, X, y):
        """
        Train the model on the provided data.

        Parameters:
        -----------
        X: Array of training data (N x M array of N samples and M features)
        y: Array of target values (N samples)

        Returns:
        --------
        self: retruns an instance of self
        """
        # Calculate the coefficients and intercept

        X, y = self.validateData(X, y)





        return self
