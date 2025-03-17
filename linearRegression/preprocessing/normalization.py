import numpy as np

class FeatureNormalizer:
    def __init__(self):
        self.mean = None
        self.std = None
        
    def fit(self, X):
        """
        Calculate the mean and standard deviation of the features.
        
        Parameters:
        -----------
        X: Array of features (N x M array of N samples and M features)
        
        Returns:
        --------
        self: an instance of self
        """
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        # Avoid division by zero
        self.std[self.std == 0] = 1
        return self
        
    def transform(self, X):
        """
        Normalize the features.
        
        Parameters:
        -----------
        X: Array of features (N x M array of N samples and M features)
        
        Returns:
        --------
        X_normalized: Normalized features
        """
        if self.mean is None or self.std is None:
            raise Exception("Normalizer has not been fitted.")
            
        return (X - self.mean) / self.std
        
    def fit_transform(self, X):
        """
        Fit and transform the features.
        
        Parameters:
        -----------
        X: Array of features (N x M array of N samples and M features)
        
        Returns:
        --------
        X_normalized: Normalized features
        """
        self.fit(X)
        return self.transform(X)
        
    def inverse_transform(self, X_normalized):
        """
        Convert normalized features back to original scale.
        
        Parameters:
        -----------
        X_normalized: Normalized features
        
        Returns:
        --------
        X: Original scale features
        """
        if self.mean is None or self.std is None:
            raise Exception("Normalizer has not been fitted.")
            
        return X_normalized * self.std + self.mean