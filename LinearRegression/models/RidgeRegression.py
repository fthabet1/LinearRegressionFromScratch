import pandas as pd
import numpy as np
from LinearRegression.models.MultivariateLinearModel import MultivariateLinearModel
from LinearRegression.optimizers.GradientDescent import GradientDescent

class RidgeRegression(MultivariateLinearModel):

    def __init__(self, max_iterations=1000, normalize=True, verbose = False, lambda_=1.0):
        super().__init__(max_iterations=max_iterations, normalize=normalize)
        self.lambda_ = lambda_
        self.verbose = verbose
        self.optimizer = GradientDescent(maxIterations=max_iterations, lambda_=self.lambda_)

    def setLambda(self, lambda_, numSamples):
        self.lambda_ = lambda_ / numSamples
        self.optimizer.setLambda(self.lambda_)

    def getLambda(self):
        return self.lambda_
    
    def fit(self, X, y, verbose=True):
        """
        Train the model on the provided data.

        Parameters:
        -----------
        X: Array of training data (N x M array of N samples and M features)
        y: Array of target values (N samples)
        verbose: Whether to print progress information

        Returns:
        --------
        self: returns an instance of self
        """
        
        self.is_pandas = isinstance(X, pd.DataFrame) or isinstance(X, pd.Series)
        self.X_columns = X.columns if isinstance(X, pd.DataFrame) else None

        X, y = self.validateData(X, y)

        if self.normalize:
            X_normalized = self.normalizer.fitTransform(X)
        else:
            X_normalized = X

        # Initial coefficients (one for each feature) and intercept
        # Initialize with least squares solution
        XTX = np.dot(X_normalized.T, X_normalized)
        XTy = np.dot(X_normalized.T, y)
        self.weights = np.linalg.solve(XTX + self.lambda_ * np.eye(X_normalized.shape[1]), XTy)
        self.bias = 0.0

        self.bias, self.weights, costHistory = self.optimizer.optimize(X_normalized, y, self.bias, self.weights)
        
        if self.verbose:
            print(f"Model trained with coefficients: {self.weights} and intercept: {self.bias}")
            print(f"Initial cost: {costHistory[0]}")
            print(f"Final cost: {costHistory[-1]}")

        return self
    
    def predict(self, X):
        """
        Make predictions using the trained model.

        Parameters:
        -----------
        X: Array of data to make predictions on (N x M array of N samples and M features)"

        Returns:
        --------
        yPred:   Array of predicted values (N predictions)
        """

        if self.weights is None or self.bias is None:
            raise Exception("Model has not been trained yet.")

        is_pandas = isinstance(X, pd.DataFrame) or isinstance(X, pd.Series)
        if is_pandas:
            originalIndex = X.index

        X, _ = self.validateData(X)

        if self.normalize:      
            X = self.normalizer.transform(X)

        predictions = np.dot(X, self.weights) + self.bias

        if is_pandas:
            return pd.Series(predictions, index=originalIndex)

        return predictions