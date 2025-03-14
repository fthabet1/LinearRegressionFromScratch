from abc import ABC, abstractmethod
import numpy as np

class BaseModel(ABC):
    
    def __init__(self):
        # These are common attributes among all regression models
        self.coefficients = None # Storing model weights
        self.intercept = None # Storing the intercept term
        self.isFitted = False # Track if the model has been trained or not

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
        self: retruns an instance of self
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
        y: Array of actual values (N smaples)

        Returns:
        --------
        score: float, R^2 score
        """

        predictions = self.predict(X)
        loss = 0

        for i in range(len(predictions)):
            loss += ((y[i] - predictions[i]) ** 2)
        

        return loss


    def validateData(self, X, y = None):
        """
        Validate and conver input data to proper numpy arrays.
        
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

        X = np.array(X)

        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        if y is not None:
            y = np.array(y)

            if len(y.shape) == 1:
                y = y.reshape(-1, 1)

            if len(y.shape) != 1:
                raise ValueError("y must be 1D")

            if X.shape[0] != y.shape[0]:
                raise ValueError("X and y must have the same number of samples")

        if np.isnan(X).any():
            raise ValueError("X contains missing values")

        if np.isinf(X).any():
            raise ValueError("X contains infinite values")

        if y is not None:
            if np.isnan(y).any():
                raise ValueError("y contains missing values")

            if np.isinf(y).any():
                raise ValueError("y contains infinite values")

        return X, y
    
    def getParams(self):
        """
        Get the parameters of the model.

        Returns:
        --------
        params: Dictionary of model parameters
        """
        return {
            "coefficients": self.coefficients,
            "intercept": self.intercept,
            "isFitted": self.isFitted
        }
    
    def setParams(self, **params):
        """
        Set the parameters of the model.

        Parameters:
        -----------
        params: Dictionary of model parameters
        """
        self.coefficients = params.get("coefficients", self.coefficients)
        self.intercept = params.get("intercept", self.intercept)
        self.isFitted = params.get("isFitted", self.isFitted)