import numpy as np

class GradientDescent:

    def __init__(self, learningRate=0.01, maxIterations=1000, tolerance=0.0001, verbose=False):
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

    def optimize(self, X, y, initialParams=None):
        """
        Optimize the model parameters using gradient descent.

        Parameters:
        -----------
        X: Array of training data (N x M array of N samples and M features)
        y: Array of target values (N samples)
        initalParams: Array of initial model parameters

        Returns:
        --------
        params: Array of optimized model parameters
        costHistory: Array of cost function history
        """
        costHistory = []

        # Type checking
        initialParams = np.array(initialParams)

        if initialParams is not None:
            params = initialParams
        else:
            params = np.ones(X.shape[1])

        for i in range(self.maxIterations):
            gradient = self.computeGradient(X, y, params)
            newParams = params - self.learningRate * gradient

            # Compute the cost function
            cost = self.computeCost(X, y, newParams)
            costHistory.append(cost)

            # Check for convergence
            if self.checkConvergence(costHistory[-1], cost):
                break

        return params, costHistory
    
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
        X = np.array(X)
        y = np.array(y)

        # Number of samples
        m = X.shape[0]

        # Calculate predictions with current parameters
        predictions = X @ params

        errors = predictions - y

        # Calculate the gradient using vectorized operations
        # (1/m) * X.T * (X*params - y)
        gradient = (1/m) * X.T @ errors

        return gradient

    def computeCost(self, X, y, params):
        """
        Compute the cost function for linear regression.
        The cost function for linear regression is the mean squared error (MSE):
        J(θ) = (1/2m) * Σ(hθ(x) - y)^2

        Parameters:
        -----------
        X: Array of training data (N x M array of N samples and M features)
        y: Array of target values (N samples)
        params: Array of model parameters

        Returns:
        --------
        cost: float, the cost of the model
        """
        X = np.array(X)
        y = np.array(y)

        # Number of samples
        m = X.shape[0]

        # Calculate predictions with current parameters
        predictions = X @ params

        # Calculate the mean squared error
        # (1/2m) * Σ(hθ(x) - y)^2
        cost = (1/(2*m)) * np.sum((predictions - y)**2)

        return cost


    def checkConvergence(self, oldCost, newCost):
        """
        Check to see if the optimizer has converged.

        Parameters:
        -----------
        oldCost: float, the cost of the model at the previous iteration
        newCost: float, the cost of the model at the current iteration

        Returns:
        --------
        converged: bool, whether the optimizer has converged
        """

        # Check if the difference in cost is less than the tolerance
        return np.abs(oldCost - newCost) < self.tolerance