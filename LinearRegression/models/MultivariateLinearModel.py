import pandas as pd
import numpy as np
from LinearRegression.models.BaseModel import BaseModel
from LinearRegression.optimizers.GradientDescent import GradientDescent
from LinearRegression.preprocessing.Normalization import FeatureNormalizer

class MultivariateLinearModel(BaseModel):

    def __init__(self, learning_rate=0.01, max_iterations=1000, normalize=True):
        super().__init__()
        self.optimizer = GradientDescent(
            learningRate=learning_rate, 
            maxIterations=max_iterations,
            tolerance=1e-8,  # Use a smaller tolerance for better convergence
            verbose=False
        )
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

        # Initialize weights with small random values instead of zeros
        # This helps avoid getting stuck in bad local minima
        self.weights = np.random.randn(X_normalized.shape[1]) * 0.01
        self.bias = 0.0
        
        # Gradient descent for specified iterations
        self.bias, self.weights, costHistory = self.optimizer.optimize(X_normalized, y, self.bias, self.weights)
        
        if verbose:
            print(f"Model trained with coefficients: {self.weights} and intercept: {self.bias}")
            print(f"Initial cost: {costHistory[0]}")
            print(f"Final cost: {costHistory[-1]}")
            
            # Print cost improvement
            cost_improvement = (costHistory[0] - costHistory[-1]) / costHistory[0] * 100
            print(f"Cost improvement: {cost_improvement:.2f}%")

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
        
        isPandas = isinstance(X, pd.DataFrame) or isinstance(X, pd.Series)
        
        # Store shape and index information before validation
        if isPandas:
            originalIndex = X.index
        
        X, _ = self.validateData(X)
        
        if self.normalize and self.normalizer is not None:
            X = self.normalizer.transform(X)
        
        # Make predictions    
        predictions = np.dot(X, self.weights) + self.bias
        
        # Ensure predictions is a 1D array
        predictions = predictions.flatten()
        
        # Return as pandas Series if input was pandas
        if isPandas:
            # Make sure the index has the same length as predictions
            if len(originalIndex) != len(predictions):
                # If lengths don't match, create a new index of appropriate length
                return pd.Series(predictions)
            else:
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
        # Convert to numpy arrays if needed
        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            y = y.values
        else:
            y = np.array(y)
            
        # Get predictions using the predict method
        predictions = self.predict(X)
        
        # Convert predictions to numpy array if it's a pandas Series
        if isinstance(predictions, pd.DataFrame) or isinstance(predictions, pd.Series):
            predictions = predictions.values
        else:
            predictions = np.array(predictions)
            
        # Make sure both are 1D arrays
        y = y.flatten()
        predictions = predictions.flatten()
        
        # Ensure lengths match
        if len(y) != len(predictions):
            # Adjust predictions to match y length
            if len(y) < len(predictions):
                predictions = predictions[:len(y)]
            else:
                # This shouldn't happen, but just in case
                raise ValueError(f"Predictions length ({len(predictions)}) is less than y length ({len(y)})")
        
        # Calculate R² score
        y_mean = np.mean(y)
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - y_mean) ** 2)
        
        if ss_tot == 0:
            return 0.0
            
        r2 = 1 - (ss_res / ss_tot)
        
        # R² sometimes can be negative if the model is worse than predicting the mean
        # In practice, it's usually capped at 0
        return max(0.0, r2) if not np.isnan(r2) else 0.0