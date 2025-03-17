'''
    The purpose of this file is to demonstrate the performance of the
    SimpleLinearModel some univariate datasets. The datasets is generated
    from Kaggle and can be found at the following links:
    https://www.kaggle.com/datasets/abhishek14398/salary-dataset-simple-linear-regression
    https://www.kaggle.com/datasets/vinaysidharth/temperature-vs-icecream-dataset
    https://www.kaggle.com/datasets/devansodariya/student-performance-data
'''
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from LinearRegression.models.SimpleLinear import SimpleLinearModel
from LinearRegression.utils.DataLoader import loadDatasetFromKaggle
from LinearRegression.preprocessing.DataSplitter import trainTestSplitData
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the datasets
salaryData = loadDatasetFromKaggle("abhishek14398/salary-dataset-simple-linear-regression")
temperatureData = loadDatasetFromKaggle("vinaysidharth/temperature-vs-icecream-dataset")
studentData = loadDatasetFromKaggle("devansodariya/student-performance-data")

# Touch up datasets for univariate analysis
studentData["Grade"] = studentData["G1"] + studentData["G2"] + studentData["G3"]
studentData["Scores"] = (studentData["Grade"] / 3).round(2)

salaryData = salaryData[["YearsExperience", "Salary"]]
temperatureData = temperatureData[["Temperature", "Ice Cream Profits"]]
studentData = studentData[["studytime", "Grade"]]

datasets = [salaryData, temperatureData, studentData]
targets = ["Salary", "Ice Cream Profits", "Grade"]
feature_cols = ["YearsExperience", "Temperature", "studytime"]

# Different learning rates for different datasets
learning_rates = [0.01, 0.001, 0.05]
max_iterations = [5000, 5000, 5000]

# Plot settings
plt.figure(figsize=(18, 12))

for i in range(len(datasets)):
    dataset = datasets[i]
    feature_col = feature_cols[i]
    target_col = targets[i]
    
    # Print dataset information
    print(f"\nDataset {i+1}: {feature_col} vs {target_col}")
    print(f"Shape: {dataset.shape}")
    print(f"Feature range: {dataset[feature_col].min()} - {dataset[feature_col].max()}")
    print(f"Target range: {dataset[target_col].min()} - {dataset[target_col].max()}")
    
    # Split the data
    X_train, X_test, y_train, y_test = trainTestSplitData(dataset, target_col)
    
    # Create and train the model with appropriate parameters
    model = SimpleLinearModel(
        learning_rate=learning_rates[i],
        max_iterations=max_iterations[i],
        normalize=True
    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    
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
    if isinstance(train_predictions, pd.Series):
        train_pred_sorted = train_predictions.iloc[X_train_sorted_idx].values
    else:
        train_pred_sorted = train_predictions[X_train_sorted_idx]
    
    plt.plot(
        X_train_sorted, 
        train_pred_sorted, 
        color='red', 
        linewidth=2, 
        label='Prediction'
    )
    
    plt.title(f'{feature_col} vs {target_col}\nR² = {test_score:.4f}')
    plt.xlabel(feature_col)
    plt.ylabel(target_col)
    plt.legend()
    
    # Plot residuals
    plt.subplot(2, 3, i+4)
    
    # Ensure residuals are calculated with matching types
    if isinstance(test_predictions, pd.Series):
        residuals = y_test.values - test_predictions.values
    else:
        residuals = y_test.values - test_predictions
    
    plt.scatter(X_test, residuals, color='purple')
    plt.axhline(y=0, color='red', linestyle='-')
    plt.title(f'Residuals plot\nMean = {np.mean(residuals):.4f}')
    plt.xlabel(feature_col)
    plt.ylabel('Residuals')
    
    print("---------------------------------------------------------")

plt.tight_layout()
plt.show()