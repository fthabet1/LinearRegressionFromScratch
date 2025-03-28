from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

class BaseModel(ABC):
    
    def __init__(self):
        # These are common attributes among all regression models
        self.weights = None  # Storing model weights
        self.bias = None  # Storing the intercept term
        self.isFitted = False  # Track if the model has been trained or not

    def getWeights(self):
        return self.weights
    
    def getBias(self):
        return self.bias
    
    def getParams(self):
        return self.weights, self.bias
    
    def setParams(self, weights, bias):
        self.weights = weights
        self.bias = bias

    @abstractmethod
    def fit(self, X, y):
        """
        Train the model on the provided data.

        Parameters:
        -----------
        X: Array of training data (N x M array of N samples and M features)
        y: Array of target values (N samples)

        Returns:
        --------
        self: returns an instance of self
        """
        pass

    @abstractmethod
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
        pass

    def score(self, X, y):
        """
        Calculate the coefficient of determination (R^2) for the model.

        Parameters:
        -----------
        X: Array of test data (N x M array of N samples and M features)
        y: Array of actual values (N samples)

        Returns:
        --------
        score: float, R^2 score
        """
        isPandas = isinstance(y, pd.DataFrame) or isinstance(y, pd.Series)
        
        if isPandas:
            y = y.values
        else:
            y = np.array(y)

        predictions = self.predict(X)
        
        if isinstance(predictions, pd.DataFrame) or isinstance(predictions, pd.Series):
            predictions = predictions.values
        else:
            predictions = np.array(predictions)
            
        # Make sure both y and predictions are 1D arrays
        y = y.flatten()
        predictions = predictions.flatten()

        y_mean = np.mean(y)
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - y_mean) ** 2)
        
        if ss_tot == 0:
            return 0.0
            
        r2 = 1 - (ss_res / ss_tot)
        
        # R² sometimes can be negative if the model is worse than predicting the mean
        # In practice, it's usually capped at 0
        return max(0.0, r2) if not np.isnan(r2) else 0.0
    

    def adjustedR2(self, X, y):
        """
        Calculate the adjusted R^2 for the model.

        Parameters:
        -----------
        X: Array of test data (N x M array of N samples and M features)
        y: Array of actual values (N samples)

        Returns:
        --------
        score: float, adjusted R^2 score
        """
        # Get the R^2 score
        r2 = self.score(X, y)
        
        # Number of samples
        n = X.shape[0]
        
        # Number of features
        p = X.shape[1]
        
        # Calculate the adjusted R^2
        adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        
        return adjusted_r2

    def crossValidation(self, X, y, k=5):
        """
        Perform k-fold cross-validation on the model.

        Parameters:
        -----------
        X: Array or DataFrame of data (N x M array of N samples and M features)
        y: Array or Series of target values (N samples)
        k: Number of folds

        Returns:
        --------
        scores: Array of R^2 scores for each fold
        """
        print(f"Starting {k}-fold cross-validation...")
        
        # Convert to numpy arrays if pandas objects
        is_pandas_X = isinstance(X, pd.DataFrame) or isinstance(X, pd.Series)
        is_pandas_y = isinstance(y, pd.DataFrame) or isinstance(y, pd.Series)
        
        X_copy, y_copy = self.validateData(X, y)
        
        n = X_copy.shape[0]
        print(f"Total samples for cross-validation: {n}")
        
        # Create random indices for shuffling
        indices = np.random.permutation(n)
        
        fold_size = n // k
        fold_indices = [indices[i * fold_size:(i + 1) * fold_size] for i in range(k)]
        
        # If n is not divisible by k, add the remaining indices to the last fold
        if n % k != 0:
            fold_indices[-1] = np.concatenate([fold_indices[-1], indices[k * fold_size:]])
        
        scores = []
        prevWeights = None
        prevBias = None
        
        for i in range(k):
            
            # Get test indices for this fold
            test_idx = fold_indices[i]
            
            # Get training indices (all indices not in test_idx)
            train_idx = np.concatenate([fold_indices[j] for j in range(k) if j != i])
            
            
            if is_pandas_X:
                X_train = X.iloc[train_idx] if isinstance(X, pd.DataFrame) else X.loc[train_idx]
                X_test = X.iloc[test_idx] if isinstance(X, pd.DataFrame) else X.loc[test_idx]
            else:
                X_train = X_copy[train_idx]
                X_test = X_copy[test_idx]
                
            if is_pandas_y:
                y_train = y.iloc[train_idx] if isinstance(y, pd.DataFrame) else y.loc[train_idx]
                y_test = y.iloc[test_idx] if isinstance(y, pd.DataFrame) else y.loc[test_idx]
            else:
                y_train = y_copy[train_idx]
                y_test = y_copy[test_idx]
            
            # Create a new instance of the same model class
            if hasattr(self, 'optimizer') and hasattr(self, 'normalize'):
                # For MultipleLinearModel
                model_copy = self.__class__(
                    max_iterations=self.optimizer.getMaxIterations() if hasattr(self.optimizer, 'getMaxIterations') else 1000,
                    normalize=self.normalize
                )
            else:
                # Generic fallback
                model_copy = self.__class__()
            
            if prevWeights is not None and prevBias is not None:
                model_copy.setParams(weights=prevWeights, bias=prevBias)


            model_copy.fit(X_train, y_train, verbose=False)
            prevWeights = model_copy.getWeights()
            prevBias = model_copy.getBias()
            
            score = model_copy.score(X_test, y_test)

            scores.append(score)
        
        print(f"Cross-validation complete. Mean R² score: {np.mean(scores):.4f}")
        return scores
    
    def validateData(self, X, y=None):
        """
        Validate and convert input data to proper numpy arrays.
        
        This function should:
        1. Convert X to a numpy array if it isn't already
        2. Ensure X is 2D (n samples, m features)
        3. If y is provided, convert it to a numpy array if it isn't already
        4. Ensure y is 1D (n samples)
        5. Check for missing or infinite values
        6. Verify X and y have the same number of samples

        Parameters:
        -----------
        X: Array of training data (N x M array of N samples and M features)
        y: Array of target values (N samples)

        Returns:
        --------
        X: Numpy array of training data
        y: Numpy array of target values (if provided) 
        """
        # Convert DataFrame or Series to numpy
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X.values
        else:
            X = np.array(X)

        # Ensure X is 2D
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        if y is not None:
            if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
                y = y.values
            else:
                y = np.array(y)
                
            # Ensure y is 1D
            y = y.flatten()

            # Check that X and y have the same number of samples
            if X.shape[0] != y.shape[0]:
                raise ValueError(f"X and y must have the same number of samples, got {X.shape[0]} and {y.shape[0]}.")

        # Check for missing or infinite values
        if not np.issubdtype(X.dtype, np.number):
            raise TypeError("X contains non-numeric data, which is not supported")
        
        if np.isnan(X).any():
            raise ValueError("X contains missing values")

        if np.isinf(X).any():
            raise ValueError("X contains infinite values")

        if y is not None:
            if np.isnan(y).any():
                raise ValueError("y contains missing values")

            if np.isinf(y).any():
                raise ValueError("y contains infinite values")

        return X, y if y is not None else None
    
    
        """
        Set the parameters of the model.

        Parameters:
        -----------
        params: Dictionary of model parameters
        """
        self.weights = params.get("coefficients", self.weights)
        self.bias = params.get("intercept", self.bias)
        self.isFitted = params.get("isFitted", self.isFitted)