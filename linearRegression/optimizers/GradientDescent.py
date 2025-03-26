import numpy as np

class GradientDescent:

    def __init__(self, learningRate=0.01, maxIterations=1000, tolerance=1e-6, verbose=False):
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

    def computeCost(self, X, y, weights, bias):
        """
        Compute the mean squared error cost.

        Parameters:
        -----------
        X: Array of training data (N x M array of N samples and M features)
        y: Array of target values (N samples)
        weights: Array of weights for each feature
        bias: Scalar bias term

        Returns:
        --------
        cost: Mean squared error cost
        """
        m = len(y)
        predictions = np.dot(X, weights) + bias
        cost = (1/(2*m)) * np.sum(np.square(predictions - y))
        return cost

    def checkConvergence(self, oldCost, newCost):
        """
        Check if the optimization has converged.

        Parameters:
        -----------
        oldCost: Previous iteration's cost
        newCost: Current iteration's cost

        Returns:
        --------
        bool: True if converged, False otherwise
        """
        return abs(oldCost - newCost) < self.tolerance

    def optimize(self, X, y, initialParams=None):
        """
        Optimize the model parameters using gradient descent.

        Parameters:
        -----------
        X: Array of training data (N x M array of N samples and M features)
        y: Array of target values (N samples)
        initialParams: Array of initial model parameters (optional)

        Returns:
        --------
        (weights, bias): Tuple of optimized weights and bias
        costHistory: Array of cost function history
        """
        X = np.array(X)
        y = np.array(y)
        m = len(y)  # number of samples
        costHistory = []

        # Initialize weights and bias
        if initialParams is not None:
            weights = np.array(initialParams[:-1], dtype=np.float64)
            bias = initialParams[-1]
        else:
            weights = np.zeros(X.shape[1], dtype=np.float64)
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
            # Make predictions
            predictions = np.dot(X, weights) + bias
            
            # Calculate errors
            errors = predictions - y
            
            # Calculate gradients
            dw = (1/m) * np.dot(X.T, errors)
            db = (1/m) * np.sum(errors)
            
            # Update parameters
            weights = weights - currentLR * dw
            bias = bias - currentLR * db
            
            # Calculate new cost
            newCost = self.computeCost(X, y, weights, bias)
            costHistory.append(newCost)
            
            # Check for convergence
            if self.checkConvergence(cost, newCost):
                if self.verbose:
                    print(f"Converged after {iteration + 1} iterations")
                break
            
            # Implement basic learning rate adjustment
            if newCost > cost:
                costIncreases += 1
                if costIncreases > 2:
                    currentLR *= 0.5
                    costIncreases = 0
                    if self.verbose:
                        print(f"Reducing learning rate to {currentLR}")
            else:
                costIncreases = 0
            
            cost = newCost
            
            if self.verbose and (iteration + 1) % 100 == 0:
                print(f"Iteration {iteration + 1}, Cost: {newCost}")
                
        return weights, bias, costHistory

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
            predictions = X @ params

            # Calculate errors
            errors = predictions - y

            # Calculate the gradient using vectorized operations
            # (1/m) * X.T * (X*params - y)
            gradient = (1/m) * X.T @ errors
            
            # Clip gradient to prevent explosion
            gradient = np.clip(gradient, -1e10, 1e10)
            
            return gradient
            
        except Exception as e:
            if self.verbose:
                print(f"Error computing gradient: {e}")
            return np.zeros_like(params)