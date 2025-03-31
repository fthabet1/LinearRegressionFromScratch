import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from LinearRegression.models.UnivariateLinearModel import UnivariateLinearModel
from LinearRegression.utils.DataLoader import loadDatasetFromCSV
from LinearRegression.preprocessing.DataSplitter import trainTestSplitData
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the datasets
salaryData = loadDatasetFromCSV("../notebooks/univariateSalaryData.csv")
iceCreamData = loadDatasetFromCSV("../notebooks/univariateIceCreamData.csv")
studentData = loadDatasetFromCSV("../notebooks/univariateStudentData.csv")

datasets = [salaryData, iceCreamData, studentData]
targets = ["Salary", "Ice Cream Profits", "Final Grade"]
feature_cols = ["YearsExperience", "Temperature", "studytime"]

# Different learning rates for different datasets
learningRates = [0.01, 0.001, 0.05]
maxIterations = [5000, 5000, 5000]

# Plot settings
plt.figure(figsize=(18, 12))

for i in range(len(datasets)):
    dataset = datasets[i]
    featureCol = feature_cols[i]
    targetCol = targets[i]
    
    # Print dataset information
    print(f"\nDataset {i+1}: {featureCol} vs {targetCol}")
    print(f"Shape: {dataset.shape}")
    print(f"Feature range: {dataset[featureCol].min()} - {dataset[featureCol].max()}")
    print(f"Target range: {dataset[targetCol].min()} - {dataset[targetCol].max()}")
    
    # Split the data
    X_train, X_test, y_train, y_test = trainTestSplitData(dataset, targetCol)
    
    # Create and train the model with appropriate parameters
    model = UnivariateLinearModel(
        learning_rate=learningRates[i],
        max_iterations=maxIterations[i],
        normalize=True
    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    trainPredictions = model.predict(X_train)
    testPredictions = model.predict(X_test)
    
    # Calculate scores
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"Model R^2 score on training data: {train_score:.4f}")
    print(f"Model R^2 score on test data: {test_score:.4f}")
    
    # Plot the results
    plt.subplot(2, 3, i+1)
    plt.scatter(X_train, y_train, color='blue', label='Training data')
    plt.scatter(X_test, y_test, color='green', label='Test data')
    
    # Convert to numpy arrays for sorting and plotting
    X_train_np = X_train.values.flatten()
    X_train_sorted_idx = np.argsort(X_train_np)
    X_train_sorted = X_train_np[X_train_sorted_idx]
    
    # For train predictions, keep the same type as the output from predict
    if isinstance(trainPredictions, pd.Series):
        sortedTrainPrediction = trainPredictions.iloc[X_train_sorted_idx].values
    else:
        sortedTrainPrediction = trainPredictions[X_train_sorted_idx]
    
    plt.plot(
        X_train_sorted, 
        sortedTrainPrediction, 
        color='red', 
        linewidth=2, 
        label='Prediction'
    )
    
    plt.title(f'{featureCol} vs {targetCol}\nR² = {test_score:.4f}')
    plt.xlabel(featureCol)
    plt.ylabel(targetCol)
    plt.legend()
    
    # Plot residuals
    plt.subplot(2, 3, i+4)
    
    # Ensure residuals are calculated with matching types
    if isinstance(testPredictions, pd.Series):
        residuals = y_test.values - testPredictions.values
    else:
        residuals = y_test.values - testPredictions
    
    plt.scatter(X_test, residuals, color='purple')
    plt.axhline(y=0, color='red', linestyle='-')
    plt.title(f'Residuals plot\nMean = {np.mean(residuals):.4f}')
    plt.xlabel(featureCol)
    plt.ylabel('Residuals')
    
    print("---------------------------------------------------------")

plt.tight_layout()
plt.show()