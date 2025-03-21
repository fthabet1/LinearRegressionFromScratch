import pandas as pd
import numpy as np
from LinearRegression.models.BaseModel import BaseModel
from LinearRegression.optimizers.GradientDescent import GradientDescent
from LinearRegression.preprocessing.Normalization import FeatureNormalizer

class SimpleLinearModel(BaseModel):

    def __init__(self, learning_rate=0.01, max_iterations=1000, normalize=True):
        super().__init__()
        self.optimizer = GradientDescent(learningRate=learning_rate, maxIterations=max_iterations)
        self.intercept = None
        self.coefficients = None
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
        
        # Normalize features if enabled
        if self.normalize:
            X_normalized = self.normalizer.fitTransform(X)
        else:
            X_normalized = X
            
        # Add bias term (column of ones)
        X_with_bias = np.c_[np.ones(X_normalized.shape[0]), X_normalized]

        # Initialize parameters with zeros
        initial_params = np.zeros(2)  # [intercept, coefficient]
        
        parameters, history = self.optimizer.optimize(X_with_bias, y, initial_params)

        self.intercept = parameters[0]
        self.coefficients = parameters[1]  # This is a scalar for simple linear regression
        self.isFitted = True

        # Print training results
        print(f"Model trained with coefficient: {self.coefficients} and intercept: {self.intercept}")
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
        if self.coefficients is None or self.intercept is None:
            raise Exception("Model has not been trained yet.")

        # Check if X is a pandas DataFrame or Series
        is_pandas_input = isinstance(X, pd.DataFrame) or isinstance(X, pd.Series)
        
        # Store original index if pandas
        original_index = X.index if is_pandas_input else None
        
        # Validate data
        X, _ = self.validateData(X)
        
        # Normalize features if model was trained with normalization
        if self.normalize:
            X = self.normalizer.transform(X)
        
        # Make predictions (ensuring 1D output)
        predictions = self.intercept + X.flatten() * self.coefficients
        
        # Return as pandas Series if input was pandas
        if is_pandas_input and original_index is not None:
            return pd.Series(predictions, index=original_index)
        
        return predictions