import pandas as pd
import numpy as np
from LinearRegression.models.BaseModel import BaseModel
from LinearRegression.optimizers.GradientDescent import GradientDescent
from LinearRegression.preprocessing.Normalization import FeatureNormalizer

class SimpleLinearModel(BaseModel):

    def __init__(self, learning_rate=0.01, max_iterations=1000, normalize=True):
        super().__init__()
        self.optimizer = GradientDescent(learningRate=learning_rate, maxIterations=max_iterations)
        self.bias = None
        self.weights = None
        self.normalize = normalize
        self.normalizer = FeatureNormalizer() if normalize else None

    def fit(self, X, y):
        """
        Train the model on the provided data.

        Parameters:
        -----------
        X: Array of training data (N x 1 array of N samples and 1 feature)
        y: Array of target values (N samples)

        Returns:
        --------
        self: returns an instance of self
        """
        self.is_pandas = isinstance(X, pd.DataFrame) or isinstance(X, pd.Series)
        self.X_columns = X.columns if isinstance(X, pd.DataFrame) else None
        
        # Validate and prepare data
        X, y = self.validateData(X, y)
        
        # Store y statistics for denormalization during prediction
        if self.normalize:
            self.y_mean = np.mean(y)
            self.y_std = np.std(y)
            X_normalized = self.normalizer.fitTransform(X)
            y_normalized = (y - self.y_mean) / self.y_std
        else:
            X_normalized = X
            y_normalized = y
        
        # For univariate regression, ensure X is properly shaped
        if X_normalized.shape[1] != 1:
            raise ValueError("SimpleLinearModel expects exactly one feature (univariate regression)")
        
        # Initialize parameters with better starting values
        # For univariate regression, we can directly compute the coefficients
        x_mean = np.mean(X_normalized)
        y_mean = np.mean(y_normalized)
        numerator = np.sum((X_normalized.flatten() - x_mean) * (y_normalized - y_mean))
        denominator = np.sum((X_normalized.flatten() - x_mean) ** 2)
        
        if denominator != 0:
            initial_weights = numerator / denominator
        else:
            initial_weights = 0.0
        
        initial_bias = y_mean - initial_weights * x_mean
        
        # Make sure the weights is a scalar for univariate regression
        if isinstance(initial_weights, np.ndarray):
            initial_weights = float(initial_weights.item()) if initial_weights.size == 1 else float(initial_weights[0])
        
        # Train the model
        self.bias, weights_array, costHistory = self.optimizer.optimize(
            X_normalized, 
            y_normalized, 
            initial_bias, 
            initial_weights
        )
        
        # Ensure weights is a scalar
        self.weights = float(weights_array) if isinstance(weights_array, np.ndarray) else weights_array
        self.isFitted = True

        # Print training results
        print(f"Model trained with coefficient: {self.weights} and intercept: {self.bias}")
        print(f"Initial cost: {costHistory[0]}")
        print(f"Final cost after {self.optimizer.getMaxIterations()} iterations: {costHistory[-1]}")

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
        if self.weights is None or self.bias is None:
            raise Exception("Model has not been trained yet.")

        # Check if X is a pandas DataFrame or Series
        isPandasInput = isinstance(X, pd.DataFrame) or isinstance(X, pd.Series)
        
        # Store original index if pandas
        originalIndex = X.index if isPandasInput else None
        
        # Validate data
        X, _ = self.validateData(X)
        
        # Normalize features if model was trained with normalization
        if self.normalize:
            X = self.normalizer.transform(X)
        
        # Ensure X is properly flattened for univariate regression
        X_flat = X.flatten()
        
        # Make predictions using scalar multiplication
        predictions = self.bias + X_flat * self.weights
        
        # Denormalize predictions if target was normalized
        if self.normalize and hasattr(self, 'y_mean') and hasattr(self, 'y_std'):
            predictions = predictions * self.y_std + self.y_mean
        
        # Return as pandas Series if input was pandas
        if isPandasInput and originalIndex is not None:
            return pd.Series(predictions, index=originalIndex)
        
        return predictions