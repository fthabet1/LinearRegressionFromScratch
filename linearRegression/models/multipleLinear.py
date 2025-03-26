import pandas as pd
import numpy as np
from LinearRegression.models.BaseModel import BaseModel
from LinearRegression.optimizers.GradientDescent import GradientDescent
from LinearRegression.preprocessing.Normalization import FeatureNormalizer

class MultipleLinearModel(BaseModel):

    def __init__(self, learning_rate=0.01, max_iterations=1000, normalize=True):
        super().__init__()
        self.optimizer = GradientDescent(learningRate=learning_rate, maxIterations=max_iterations)
        self.bias = None
        self.weights = None
        self.normalize = normalize
        self.normalizer = FeatureNormalizer() if normalize else None
    
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
        
        # Gradient descent for specified iterations
        self.weights, self.bias, costHistory = self.optimizer.optimize(X_normalized, y)
        
        if verbose:
            print(f"Model trained with coefficients: {self.weights} and intercept: {self.bias}")
            print(f"Initial cost: {costHistory[0]}")
            print(f"Final cost: {costHistory[-1]}")

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
        
        # Handle pandas input
        is_pandas = isinstance(X, pd.DataFrame) or isinstance(X, pd.Series)
        if is_pandas:
            originalIndex = X.index
            
        # Convert pandas to numpy if necessary
        X, _ = self.validateData(X)
        
        # Apply normalization if it was used during training
        if self.normalize and self.normalizer is not None:
            X = self.normalizer.transform(X)
        
        # Make predictions    
        predictions = np.dot(X, self.weights) + self.bias
        
        # Return as pandas Series if input was pandas
        if is_pandas:
            return pd.Series(predictions, index=originalIndex)
        
        return predictions
        
    def score(self, X, y):
        """
        Calculate the R² score for the model.
        
        Parameters:
        -----------
        X: Array of test data
        y: Array of target values
        
        Returns:
        --------
        score: The R² score
        """
        # Ensure prediction uses same preprocessing as training
        return super().score(X, y)