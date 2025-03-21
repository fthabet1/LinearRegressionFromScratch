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
        X = np.array(X)
        y = np.array(y)
        costHistory = []

        # Type checking and initialization
        if initialParams is not None:
            params = np.array(initialParams, dtype=np.float64)
        else:
            params = np.zeros(X.shape[1], dtype=np.float64)

        # Calculate initial cost
        cost = self.computeCost(X, y, params)
        costHistory.append(cost)
        
        if self.verbose:
            print(f"Initial cost: {cost}")

        # Current learning rate - may be adjusted
        currentLR = self.learningRate
        
        # Keep track of consecutive increases in cost
        costIncreases = 0
        
        # Gradient descent iterations
        for i in range(self.maxIterations):
            gradient = self.computeGradient(X, y, params)
            
            # Check for NaN or Inf in gradient
            if np.any(np.isnan(gradient)) or np.any(np.isinf(gradient)):
                if self.verbose:
                    print(f"Warning: NaN or Inf detected in gradient at iteration {i}")
                
                # Reduce learning rate significantly and reset parameters
                currentLR *= 0.1
                if currentLR < 1e-10:
                    if self.verbose:
                        print("Learning rate too small, stopping optimization")
                    break
                
                continue
            
            # Calculate new parameters
            newParams = params - currentLR * gradient
            
            # Calculate new cost
            try:
                newCost = self.computeCost(X, y, newParams)
                
                # Adaptive learning rate
                if self.adaptive_lr:
                    if newCost > cost:
                        # Cost increased, reduce learning rate
                        currentLR *= 0.5
                        costIncreases += 1
                        
                        if costIncreases > 5:
                            if self.verbose:
                                print(f"Too many cost increases, stopping at iteration {i}")
                            break
                            
                        # Try again with smaller learning rate
                        continue
                    else:
                        # Cost decreased, slightly increase learning rate
                        currentLR *= 1.05
                        costIncreases = 0
                
                # Check for convergence - relative improvement
                if self.checkConvergence(cost, newCost):
                    if self.verbose:
                        print(f"Converged after {i+1} iterations")
                    break
                
                # Update parameters and cost
                params = newParams
                cost = newCost
                costHistory.append(cost)
                
                if self.verbose and i % 100 == 0:
                    print(f"Iteration {i}, cost: {cost}, learning rate: {currentLR}")
                    
            except (RuntimeWarning, OverflowError, FloatingPointError) as e:
                if self.verbose:
                    print(f"Numerical error at iteration {i}: {e}")
                # Reduce learning rate
                currentLR *= 0.1
                if currentLR < 1e-10:
                    break
                continue

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
        try:
            # Number of samples
            m = X.shape[0]

            # Calculate predictions with current parameters
            predictions = X @ params

            # Check for overflow in squared error calculation
            errors = predictions - y
            squared_errors = np.square(errors)
            
            if np.any(np.isinf(squared_errors)):
                return float('inf')
                
            # Calculate the mean squared error
            # (1/2m) * Σ(hθ(x) - y)^2
            cost = (1/(2*m)) * np.sum(squared_errors)
            
            return cost
            
        except Exception as e:
            print(f"Error computing cost: {e}")
            return float('inf')

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
        if oldCost == 0:
            return abs(newCost) < self.tolerance
        return abs((newCost - oldCost) / oldCost) < self.tolerance