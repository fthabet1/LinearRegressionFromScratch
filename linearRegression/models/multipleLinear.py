import pandas as pd
import numpy as np
from LinearRegression.models.BaseModel import BaseModel
from LinearRegression.optimizers.GradientDescent import GradientDescent

class MultipleLinearModel(BaseModel):

    def __init__(self):
        super().__init__()
        self.optimizer = GradientDescent()
        self.coefficients = None
        self.intercept = None
    
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
        X, y = self.validateData(X, y)

        # Initial coefficients and intercept
        self.coefficients = np.ones(X.shape[1])
        self.intercept = 1

        # Train the model
        parameters, history = self.optimizer.optimize(X, y, self.coefficients, self.intercept)

        # Assign the coefficients and intercept
        self.coefficients = parameters[:-1]
        self.intercept = parameters[-1]

        print(f"Model trained with coefficients: {self.coefficients} and intercept: {self.intercept}")
        print(f"Initial cost: {history[0]}")
        print(f"Final cost after {self.optimizer.getMaxIterations()} iterations: {history[-1]}")

        return self
    
    def predict(self, X):
        """
        Make predictions using the trained model.

        Parameters:
        -----------
        X: Array of data to make predictions on (N x M array of N samples and M features)

        Returns:
        --------
        yPred: Array of predicted values (N predictions)
        """
        if self.coefficients is None or self.intercept is None:
            raise Exception("Model has not been trained yet.")
        
        return np.dot(X, self.coefficients) + self.intercept