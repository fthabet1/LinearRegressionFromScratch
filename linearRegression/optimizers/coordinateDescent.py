import numpy as np

class CoordinateDescent:

    def __init__(self, maxIterations = 1000, tolerance = 1e-6, verbose = True, lambda_ = 1.0):
        """
        Initialize the coordinate descent optimizer.
        
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

        self.maxIterations = maxIterations
        self.tolerance = tolerance
        self.verbose = verbose
        self.lambda_ = lambda_

    def getMaxIterations(self):
        return self.maxIterations

    def setMaxIterations(self, maxIterations):
        self.maxIterations = maxIterations

    def getTolerance(self):
        return self.tolerance

    def setTolerance(self, tolerance):
        self.tolerance = tolerance

    def getVerbose(self):
        return self.verbose

    def setVerbose(self, verbose):
        self.verbose = verbose

    def getLambda(self):
        return self.lambda_

    def setLambda(self, lambda_):
        self.lambda_ = lambda_

    def optimize(self, X, y):
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

        nSamples = X.shape[0]
        mFeatures = X.shape[1]
        
        # Clip very large values to prevent numerical issues
        theta = np.zeros(mFeatures)
        yPred = np.zeros(nSamples)

        bias = np.mean(y)
        yPred = np.dot(X, theta) + bias

        cost = self.computeCost(X, y, theta, bias)
        costHistory.append(cost)
        
        costIncreases = 0

        for i in range(self.maxIterations):
            thetaOld = theta.copy()

            # Loop through each feature and update it
            for j in range(mFeatures):
                # Store old parameter value
                thetaJOld = theta[j]
            
                theta[j] = self.computeCoordinate(X, y, theta, j, yPred)
                
                # Update predictions if the parameter changed
                if theta[j] != thetaJOld:
                    # Update predictions incrementally
                    yPred -= X[:, j] * thetaJOld
                    yPred += X[:, j] * theta[j]
            
            newCost = self.computeCost(X, y, theta, bias)
            costHistory.append(newCost)
            
            if newCost > cost:
                costIncreases += 1
                if costIncreases >= 5:
                    if self.verbose:
                        print(f"Cost increased {costIncreases} times. Exiting...")
                    break
            else:
                costIncreases = 0
        
            # Check for convergence - either by cost improvement or parameter change
            if self.checkConvergence(cost, newCost, theta, thetaOld):
                if self.verbose:
                    print(f"Converged in {i+1} iterations")
                break

            cost = newCost

        return theta, bias, costHistory


    def computeCoordinate(self, X, y, params, j, predictions=None):
        """
        Compute the coordinate descent update for a single parameter.

        Parameters:
        -----------
        X: Array of training data (N x M array of N samples and M features)
        y: Array of target values (N samples)
        params: Array of model parameters
        j: int, the index of the parameter to update
        predictions: Array of current predictions (optional, for efficiency)

        Returns:
        --------
        newParam: float, the updated parameter value
        """
        N = X.shape[0]
    
        # Get the j-th feature vector
        xj = X[:, j]
        
        # Current value of parameter j
        thetaJOld = params[j]
        
        if predictions is not None:
            residuals = y - predictions
            partialResiduals = residuals + xj * thetaJOld
        else:
            paramsWithoutJ = params.copy()
            paramsWithoutJ[j] = 0
            predsWithoutJ = np.dot(X, paramsWithoutJ)
            
            partialResiduals = y - predsWithoutJ
        
        # Calculate correlation of feature j with partial residuals
        rho = np.dot(xj, partialResiduals) / N
        
        # Apply soft thresholding
        if rho < -self.lambda_ / N:
            newParam = rho + self.lambda_ / N
        elif rho > self.lambda_ / N:
            newParam = rho - self.lambda_ / N
        else:
            newParam = 0.0
            
        return newParam


    def computeCost(self, X, y, theta, bias):
        """
        Calculate the cost function.

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
            N = X.shape[0]
            predictions = np.dot(X, theta) + bias
            residuals = y - predictions
            cost = (1/(2*N)) * np.sum(residuals**2) + (self.lambda_/N) * np.sum(np.abs(theta))
            return cost
        except Exception as e:
            print(f"Error in computeCost: {e}")
            return float('inf')
        
        
    def checkConvergence(self, oldCost, newCost, theta, thetaOld):
        """
        Check convergence for Lasso regression.
        
        Parameters:
        -----------
        oldCost: float, previous iteration's cost
        newCost: float, current iteration's cost
        theta: array, current parameters
        thetaOld: array, previous parameters
        
        Returns:
        --------
        bool: True if converged, False otherwise
        """
        # Check relative improvement in objective
        if oldCost == 0:
            obj_improvement = abs(newCost)
        else:
            obj_improvement = abs((newCost - oldCost) / oldCost)
        
        # Check parameter changes
        param_change = np.max(np.abs(theta - thetaOld))
        
        # Convergence criteria:
        # 1. Objective improvement is small
        # 2. Parameter changes are small
        # 4. Minimum improvement threshold is met
        return (obj_improvement < self.tolerance and 
                param_change < self.tolerance)