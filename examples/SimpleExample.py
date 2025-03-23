import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from LinearRegression.models.MultipleLinear import MultipleLinearModel

# Create synthetic data with known relationships
np.random.seed(42)
n_samples = 100
n_features = 3

# Create predictors with some correlation
X = np.random.randn(n_samples, n_features)

# Create targets with known coefficients: y = 2*X1 - 1*X2 + 0.5*X3 + 3 + noise
true_coefficients = np.array([2.0, -1.0, 0.5])
true_intercept = 3.0
noise = np.random.randn(n_samples) * 0.5

y = X.dot(true_coefficients) + true_intercept + noise

# Train the model
model = MultipleLinearModel(learning_rate=0.1, max_iterations=1000, normalize=True)
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Calculate and print R² score
r2_score = model.score(X, y)
print(f"R² score: {r2_score:.4f}")

# Print true vs. predicted coefficients
print(f"\nTrue coefficients: {true_coefficients}")
print(f"True intercept: {true_intercept}")
print(f"Learned coefficients: {model.weights}")
print(f"Learned intercept: {model.bias}")

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y, y_pred, alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.title('Actual vs. Predicted Values')
plt.grid(True)
plt.show() 