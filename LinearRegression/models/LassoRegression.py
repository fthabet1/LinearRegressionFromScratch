import pandas as pd
import numpy as np
from LinearRegression.models.MultivariateLinearModel import MultivariateLinearModel
from LinearRegression.optimizers.CoordinateDescent import CoordinateDescent

class LassoRegression(MultivariateLinearModel):

    def __init__(self, max_iterations=1000, normalize=True, verbose = False):
        super().__init__(max_iterations=max_iterations, normalize=normalize)
        self.lambda_ = 1.0
        self.verbose = verbose
        self.optimizer = CoordinateDescent(maxIterations=max_iterations, lambda_=self.lambda_)

    def setLambda(self, lambda_):
        self.lambda_ = lambda_
        self.optimizer.setLambda(lambda_)

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
        self.weights = np.zeros(X_normalized.shape[1])
        self.bias = 0.0
        
        # Track cost history
        cost_history = []
        
        # Number of samples and features
        m_samples = X_normalized.shape[0]
        n_features = X_normalized.shape[1]

        self.weights, self.bias, cost_history = self.optimizer.optimize(X_normalized, y)
        
        if self.verbose:
            print(f"Model trained with coefficients: {self.weights} and intercept: {self.bias}")
            print(f"Initial cost: {cost_history[0]}")
            print(f"Final cost: {cost_history[-1]}")

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

        if self.weights is None or self.bias is None:
            raise Exception("Model has not been trained yet.")
        
        is_pandas = isinstance(X, pd.DataFrame) or isinstance(X, pd.Series)
        if is_pandas:
            originalIndex = X.index

        X, _ = self.validateData(X)

        if self.normalize:
            X_normalized = self.normalizer.fitTransform(X)
        else:
            X_normalized = X

        predictions = np.dot(X_normalized, self.weights) + self.bias

        if is_pandas:
            return pd.Series(predictions, index=originalIndex)

        return predictions
    
    