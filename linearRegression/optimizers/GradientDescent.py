import numpy as np

class GradientDescent:

    def __init__(self, learningRate=0.01, maxIterations=1000, tolerance=1e-6, verbose=False, lambda_=0.0):
        """
        Initialize the gradient descent optimizer.
        
        Parameters:
        -----------
        learningRate: float, the step size to take in the direction of the gradient
        maxIterations: int, the maximum number of iterations to run
        tolerance: float, convergence threshold for stopping
        verbose: bool, whether to print out the loss at each iteration

        Returns:
        --------
        self: an instance of self
        """

        self.learningRate = learningRate
        self.maxIterations = maxIterations
        self.tolerance = tolerance
        self.verbose = verbose
        self.adaptive_lr = True
        self.lambda_ = lambda_
        self.min_lr = 1e-10  # Minimum learning rate

    def getLearningRate(self):
        return self.learningRate
    
    def getMaxIterations(self):
        return self.maxIterations
    
    def getTolerance(self):
        return self.tolerance

    def setLearningRate(self, learningRate):
        self.learningRate = learningRate

    def setMaxIterations(self, maxIterations):
        self.maxIterations = maxIterations

    def setTolerance(self, tolerance):
        self.tolerance = tolerance

    def setVerbose(self, verbose):
        self.verbose = verbose

    def setLambda(self, lambda_):
        self.lambda_ = lambda_

    def getLambda(self):
        return self.lambda_

    def computeCost(self, X, y, weights, bias):
        """
        Compute the mean squared error cost.

        Parameters:
        -----------
        X: Array of training data (N x M array of N samples and M features)
        y: Array of target values (N samples)
        weights: Array of weights for each feature or scalar for univariate
        bias: Scalar bias term

        Returns:
        --------
        cost: Mean squared error cost
        """
        m = len(y)
        
        # Handle both univariate (scalar weights) and multivariate cases
        if np.isscalar(weights) and self.lambda_ == 0.0:
            # For univariate regression with scalar weight
            X_flat = X.flatten()
            predictions = bias + X_flat * weights
        else:
            # For multivariate regression with weight vector
            predictions = np.dot(X, weights) + bias

        # Calculate regularization term
        regularizationTerm = 0.0
        if self.lambda_ != 0:
            if np.isscalar(weights):
                regularizationTerm = self.lambda_ * (weights ** 2)
            else:
                regularizationTerm = self.lambda_ * np.sum(np.square(weights))

        # Calculate cost
        cost = (1/(2*m)) * np.sum(np.square(y - predictions)) + regularizationTerm
        return cost

    def checkConvergence(self, oldCost, newCost, weight, weightOld):
        """
        Check if the optimization has converged.
        
        Parameters:
        -----------
        oldCost: Previous cost
        newCost: Current cost
        weight: Current weights (can be scalar or array)
        weightOld: Previous weights (can be scalar or array)
        
        Returns:
        --------
        bool: True if converged, False otherwise
        """
        if oldCost == 0:
            obj_improvement = abs(newCost)
        else:
            obj_improvement = abs((newCost - oldCost) / oldCost)
        
        # Handle both scalar and array weights
        if np.isscalar(weight):
            param_change = abs(weight - weightOld)
        else:
            param_change = np.max(np.abs(weight - weightOld))
        
        return (obj_improvement < self.tolerance and 
                param_change < self.tolerance)

    def optimize(self, X, y, bias=None, weights=None):
        """
        Optimize the model parameters using gradient descent.
        
        Parameters:
        -----------
        X: Array of training data (N x M array of N samples and M features)
        y: Array of target values (N samples)
        bias: Initial bias term (optional)
        weights: Initial weights (optional, can be scalar for univariate or array for multivariate)
        
        Returns:
        --------
        bias: Optimized bias term
        weights: Optimized weights (scalar or array)
        costHistory: History of cost values
        """
        X = np.array(X)
        y = np.array(y)
        m = len(y)  # number of samples
        costHistory = []
        
        # Determine if we're doing univariate or multivariate regression
        is_univariate = X.shape[1] == 1
        
        # Initialize weights and bias if not provided
        if weights is None:
            if is_univariate:
                weights = 0.0  # Scalar for univariate
            else:
                weights = np.zeros(X.shape[1], dtype=np.float64)  # Array for multivariate
        
        if bias is None:
            bias = 0.0

        # Calculate initial cost
        cost = self.computeCost(X, y, weights, bias)
        costHistory.append(cost)
        
        if self.verbose:
            print(f"Initial cost: {cost}")

        # Current learning rate - may be adjusted
        currentLR = self.learningRate
        
        # Keep track of consecutive increases in cost
        costIncreases = 0
        
        # Gradient descent iterations
        for iteration in range(self.maxIterations):
            # Store old weights for convergence check
            if np.isscalar(weights):
                weightsOld = weights
            else:
                weightsOld = weights.copy()
            
            # Calculate predictions and errors
            if is_univariate:
                X_flat = X.flatten()
                predictions = bias + X_flat * weights
                errors = predictions - y
                
                # Calculate gradients for univariate case
                dw = (1/m) * np.sum(errors * X_flat)
            else:
                predictions = np.dot(X, weights) + bias
                errors = predictions - y
                
                # Calculate gradients for multivariate case
                dw = (1/m) * np.dot(X.T, errors)
            
            # Common gradient for bias
            db = (1/m) * np.sum(errors)
            
            # Add regularization term to weight gradient if lambda is non-zero
            if self.lambda_ != 0:
                if np.isscalar(weights):
                    dw += (self.lambda_ / m) * weights
                else:
                    dw += (self.lambda_ / m) * weights
            
            # Update parameters
            weights = weights - currentLR * dw
            bias = bias - currentLR * db
            
            # Calculate new cost
            newCost = self.computeCost(X, y, weights, bias)
            costHistory.append(newCost)
            
            # Check for convergence
            if self.checkConvergence(cost, newCost, weights, weightsOld):
                if self.verbose:
                    print(f"Converged after {iteration + 1} iterations")
                break
            
            # Implement learning rate adjustment
            if newCost > cost:
                costIncreases += 1
                if costIncreases > 2:
                    # Reduce learning rate
                    currentLR *= 0.5
                    costIncreases = 0
                    if self.verbose:
                        print(f"Reducing learning rate to {currentLR}")
                    
                    # Check if learning rate is too small
                    if currentLR < self.min_lr:
                        if self.verbose:
                            print(f"Learning rate too small. Stopping.")
                        break
            else:
                costIncreases = 0
            
            cost = newCost
            
            if self.verbose and (iteration + 1) % 100 == 0:
                print(f"Iteration {iteration + 1}, Cost: {newCost}")
                
        return bias, weights, costHistory

    def computeGradient(self, X, y, params):
        """
        Computer the gradient of the cost function
        For linear regression with MSE loss, the gradient is:
        ∇J(θ) = (1/m) * X^T * (X*θ - y)

        Parameters:
        -----------
        X: Array of training data (N x M array of N samples and M features)
        y: Array of target values (N samples)
        params: Array of model parameters

        Returns:
        --------
        gradient: Array of gradients
        """
        try:
            # Number of samples
            m = X.shape[0]

            # Calculate predictions with current parameters
            if np.isscalar(params):
                X_flat = X.flatten()
                predictions = X_flat * params
                gradient = (1/m) * np.sum((predictions - y) * X_flat)
            else:
                predictions = X @ params
                # Calculate the gradient using vectorized operations
                gradient = (1/m) * X.T @ (predictions - y)
            
            # Clip gradient to prevent explosion
            if np.isscalar(gradient):
                gradient = np.clip(gradient, -1e10, 1e10)
            else:
                gradient = np.clip(gradient, -1e10, 1e10)
            
            return gradient
            
        except Exception as e:
            if self.verbose:
                print(f"Error computing gradient: {e}")
            if np.isscalar(params):
                return 0.0
            else:
                return np.zeros_like(params)