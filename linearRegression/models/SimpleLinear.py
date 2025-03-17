import pandas as pd
import numpy as np
from linearRegression.models.BaseModel import BaseModel
from linearRegression.optimizers.GradientDescent import GradientDescent

class SimpleLinearModel(BaseModel):

    def __init__(self):
        super().__init__()
        self.optimizer = GradientDescent()
        self.intercept = None
        self.coefficient = None

    def fit(self, X, y):
        """
        Train the model on the provided data.

        Parameters:
        -----------
        X: Array of training data (N x 1 array of N samples and 1 feature)
        y: Array of target values (N samples)

        Returns:
        --------
        self: retruns an instance of self
        """
        X, y = self.validateData(X, y)

        # Initial coefficients and intercept
        self.coefficient = np.ones(X.shape[1])
        self.intercept = 1

        # Train the model
        parameters, history = self.optimizer.optimize(X, y, self.coefficient, self.intercept)

        self.coefficient = parameters[0]
        self.intercept = parameters[1]

        print(f"Model trained with coefficient: {self.coefficient} and intercept: {self.intercept}")
        print(f"Initial cost: {history[0]}")
        print(f"Final cost after {self.optimizer.getMaxIterations()} iterations: {history[-1]}")

        return self

    def predict(self, X):
        """
        Make predictions using the trained model.

        Parameters:
        -----------
        X: Array of data to make predictions on (N x 1 array of N samples and 1 feature)

        Returns:
        --------
        yPred: Array of predicted values (N predictions)
        """
        if self.coefficient is None or self.intercept is None:
            raise Exception("Model has not been trained yet.")

        return np.dot(X, self.coefficient) + self.intercept