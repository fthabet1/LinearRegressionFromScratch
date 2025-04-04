class FeatureNormalizer {
    constructor() {
        this.mean = null;
        this.std = null;
    }

    fit(X) {
        if (Array.isArray(X[0])) {
            // Multivariate case
            this.mean = X[0].map((_, colIndex) => 
                X.reduce((sum, row) => sum + row[colIndex], 0) / X.length
            );
            this.std = X[0].map((_, colIndex) => {
                const colMean = this.mean[colIndex];
                const squaredDiffs = X.map(row => Math.pow(row[colIndex] - colMean, 2));
                return Math.sqrt(squaredDiffs.reduce((a, b) => a + b) / X.length);
            });
        } else {
            // Univariate case
            this.mean = X.reduce((a, b) => a + b) / X.length;
            const squaredDiffs = X.map(x => Math.pow(x - this.mean, 2));
            this.std = Math.sqrt(squaredDiffs.reduce((a, b) => a + b) / X.length);
        }
    }

    transform(X) {
        if (Array.isArray(X[0])) {
            // Multivariate case
            return X.map(row => 
                row.map((val, j) => (val - this.mean[j]) / this.std[j])
            );
        } else {
            // Univariate case
            return X.map(x => (x - this.mean) / this.std);
        }
    }

    inverse_transform(X) {
        if (Array.isArray(X[0])) {
            // Multivariate case
            return X.map(row => 
                row.map((val, j) => val * this.std[j] + this.mean[j])
            );
        } else {
            // Univariate case
            return X.map(x => x * this.std + this.mean);
        }
    }
}

class BaseModel {
    constructor() {
        this.weights = null;
        this.bias = null;
        this.normalize = false;
    }

    async fit(X, y) {
        // Validate inputs
        if (!X || !y || X.length === 0 || y.length === 0 || X.length !== y.length) {
            throw new Error("Invalid input data");
        }

        try {
            // Train the model using the API - send the raw data directly
            // The Python model will handle normalization internally
            const response = await fetch('/api/train', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    modelType: this.modelType,
                    X: X,
                    y: y,
                    learning_rate: this.learning_rate,
                    max_iterations: this.max_iterations,
                    params: this.getParams()
                })
            });

            // Log the request data for debugging
            console.log('Train request data:', {
                modelType: this.modelType,
                X_shape: `${X.length} samples x ${X[0].length || 1} features`,
                y_length: y.length,
                learning_rate: this.learning_rate,
                max_iterations: this.max_iterations,
                params: this.getParams()
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            if (!result.success) {
                throw new Error(result.error || 'Training failed');
            }

            this.weights = result.weights;
            this.bias = result.bias;
            
            return this;
        } catch (error) {
            console.error('Training error:', error);
            throw error;
        }
    }

    async predict(X) {
        if (!this.weights || this.bias === null) {
            throw new Error("Model not trained yet");
        }

        try {
            // Make predictions using the API - send raw data directly
            // The Python model will handle normalization internally
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    modelType: this.modelType,
                    X: X
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            if (!result.success) {
                throw new Error(result.error || 'Prediction failed');
            }

            return result.predictions;
        } catch (error) {
            console.error('Prediction error:', error);
            throw error;
        }
    }

    async score(X, y) {
        if (!this.weights || this.bias === null) {
            throw new Error("Model not trained yet");
        }

        try {
            // Calculate R² score using the API - send raw data directly
            // The Python model will handle normalization internally
            const response = await fetch('/api/score', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    modelType: this.modelType,
                    X: X,
                    y: y
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            if (!result.success) {
                throw new Error(result.error || 'Score calculation failed');
            }

            return result.score;
        } catch (error) {
            console.error('Score calculation error:', error);
            throw error;
        }
    }

    getParams() {
        return {};
    }
}

class UnivariateLinearModel extends BaseModel {
    constructor(learning_rate = 0.01, max_iterations = 1000) {
        super();
        this.modelType = 'univariate';
        this.learning_rate = learning_rate;
        this.max_iterations = max_iterations;
    }
}

class MultivariateLinearModel extends BaseModel {
    constructor(learning_rate = 0.01, max_iterations = 1000) {
        super();
        this.modelType = 'multivariate';
        this.learning_rate = learning_rate;
        this.max_iterations = max_iterations;
    }
}

class RidgeRegression extends BaseModel {
    constructor(learning_rate = 0.01, max_iterations = 1000, lambda = 1.0) {
        super();
        this.modelType = 'ridge';
        this.learning_rate = learning_rate;
        this.max_iterations = max_iterations;
        this.lambda = lambda;
    }

    getParams() {
        return {
            lambda: this.lambda
        };
    }
}

class LassoRegression extends BaseModel {
    constructor(learning_rate = 0.01, max_iterations = 1000, lambda = 1.0) {
        super();
        this.modelType = 'lasso';
        this.learning_rate = learning_rate;
        this.max_iterations = max_iterations;
        this.lambda = lambda;
    }

    getParams() {
        return {
            lambda: this.lambda
        };
    }
} 