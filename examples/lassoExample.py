import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from LinearRegression.models.LassoRegression import LassoRegression
from LinearRegression.utils.DataLoader import loadDatasetFromCSV
from LinearRegression.preprocessing.DataSplitter import trainTestSplitData
from LinearRegression.preprocessing.Normalization import FeatureNormalizer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

# Load the datasets
studentData = loadDatasetFromCSV("../notebooks/multivariateStudentData.csv")
housingData = loadDatasetFromCSV("../notebooks/multivariateHousingData.csv")
carPriceData = loadDatasetFromCSV("../notebooks/multivariateCarPricesData.csv")

maxIterations = 5000
k_folds = 5  # Number of folds for cross-validation

datasets = [studentData, housingData, carPriceData]
targets = ["Performance Index", "median_house_value", "selling_price"]

# Plot settings
plt.figure(figsize=(15, 15))

# Lambda values to try
lambda_values = [0.0001, 0.001, 0.01, 0.1, 1.0]

for i in range(len(datasets)):
    dataset = datasets[i]
    targetCol = targets[i]

    # Print dataset information
    print(f"\n{'='*50}")
    print(f"DATASET {i+1}: {targetCol}")
    print(f"{'='*50}")
    print(f"Shape: {dataset.shape}")
    print(f"Number of features: {dataset.shape[1] - 1}")
    print(f"Target range: {dataset[targetCol].min()} - {dataset[targetCol].max()}")
    
    # Prepare data
    X = dataset.drop(targetCol, axis=1)
    y = dataset[targetCol]
    
    # Normalize target variable
    y_normalizer = FeatureNormalizer()
    y_normalized = y_normalizer.fitTransform(y.values.reshape(-1, 1)).flatten()
    
    # 1. CROSS-VALIDATION EVALUATION
    print(f"\n1. CROSS-VALIDATION RESULTS")
    print(f"{'-'*30}")
    
    # Perform k-fold cross-validation for each lambda
    bestLambda = None
    bestCV_score = float('-inf')
    
    for lambda_ in lambda_values:
        print(f"\nTrying lambda = {lambda_}")
        
        # Create model for cross-validation
        CV_model = LassoRegression(
            max_iterations=maxIterations,
            normalize=True
        )
        CV_model.setLambda(lambda_)
        
        # Run cross-validation
        CV_score = CV_model.crossValidation(X, y_normalized, k=k_folds)
        meanCV_score = np.mean(CV_score)
        
        print(f"Mean CV R² score: {meanCV_score:.4f}")
        
        if meanCV_score > bestCV_score:
            bestCV_score = meanCV_score
            bestLambda = lambda_
    
    print(f"\nBest lambda: {bestLambda}")
    print(f"Best CV R² score: {bestCV_score:.4f}")
    
    # 2. TRAIN/TEST SPLIT EVALUATION
    print(f"\n2. TEST SET RESULTS")
    print(f"{'-'*30}")
    
    # Split the data
    X_train, X_test, y_train, y_test = trainTestSplitData(dataset, targetCol)
    print(f"Train set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")

    # Normalize target variables
    yTrainNormalized = y_normalizer.transform(y_train.values.reshape(-1, 1)).flatten()
    yTestNormalized = y_normalizer.transform(y_test.values.reshape(-1, 1)).flatten()

    # Create and train model with best lambda
    model = LassoRegression(
        max_iterations=maxIterations,
        normalize=True
    )
    model.setLambda(bestLambda)
    
    # Train and evaluate
    model.fit(X_train, yTrainNormalized, verbose=False)

    # Get predictions and denormalize them
    testPredictions = model.predict(X_test)
    testPredictions = y_normalizer.inverseTransform(testPredictions.values.reshape(-1, 1)).flatten()

    # Calculate metrics on original scale
    test_score = 1 - np.sum((y_test.values - testPredictions) ** 2) / np.sum((y_test.values - np.mean(y_test.values)) ** 2)
    test_mse = np.mean((y_test.values - testPredictions) ** 2)
    
    # Print test results
    print(f"TEST SET R² SCORE: {test_score:.4f}")
    print(f"TEST SET MSE: {test_mse:.4f}")
    
    # Print feature importance
    # feature_importance = pd.DataFrame({
    #     'Feature': X.columns,
    #     'Coefficient': model.weights
    # })
    # feature_importance = feature_importance.sort_values('Coefficient', key=abs, ascending=False)
    # print(f"Features and their coefficients:\n {feature_importance}")
    # print("\nTop 5 most important features:")
    # print(feature_importance.head())
    
    print("---------------------------------------------------------")





