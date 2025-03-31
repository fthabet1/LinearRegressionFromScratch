import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from LinearRegression.models.MultivariateLinearModel import MultivariateLinearModel
from LinearRegression.utils.DataLoader import loadDatasetFromCSV
from LinearRegression.preprocessing.DataSplitter import trainTestSplitData
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
    
    # 1. CROSS-VALIDATION EVALUATION
    print(f"\n1. CROSS-VALIDATION RESULTS")
    print(f"{'-'*30}")
    
    # Prepare data for cross-validation
    X = dataset.drop(targetCol, axis=1)
    y = dataset[targetCol]
    
    # Perform k-fold cross-validation
    print(f"Performing {k_folds}-fold cross-validation:")
    
    # Create model for cross-validation
    cv_model = MultivariateLinearModel(
        learning_rate=0.05,
        max_iterations=5000,
        normalize=True
    )
    
    # Run cross-validation
    cv_scores = cv_model.crossValidation(X, y, k=k_folds)
    
    # Print cross-validation results
    print(f"\nCross-validation R² scores:")
    for j, score in enumerate(cv_scores):
        print(f"Fold {j+1}: {score:.4f}")
    
    mean_cv_score = np.mean(cv_scores)
    std_cv_score = np.std(cv_scores)
    print(f"Mean CV R² score: {mean_cv_score:.4f} (±{std_cv_score:.4f})")
    
    # 2. TRAIN/TEST SPLIT EVALUATION
    print(f"\n2. TEST SET RESULTS")
    print(f"{'-'*30}")
    
    # Split the data
    X_train, X_test, y_train, y_test = trainTestSplitData(dataset, targetCol)
    print(f"Train set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")

    # Create and train model
    model = MultivariateLinearModel(
        learning_rate=0.05,
        max_iterations=5000,
        normalize=True
    )
    
    # Train and evaluate
    model.fit(X_train, y_train, verbose=False)
    test_predictions = model.predict(X_test)
    test_score = model.score(X_test, y_test)
    
    # Calculate MSE
    if isinstance(test_predictions, pd.Series):
        # Ensure both arrays are the same length
        if len(y_test) != len(test_predictions):
            if len(y_test) < len(test_predictions):
                test_predictions = test_predictions.iloc[:len(y_test)]
            else:
                raise ValueError(f"Predictions length less than test set length")
        test_mse = np.mean((y_test.values - test_predictions.values) ** 2)
    else:
        # Convert to numpy arrays and ensure same length
        y_test_np = y_test.values if isinstance(y_test, pd.Series) else y_test
        test_predictions_np = np.array(test_predictions)
        
        # Ensure both are 1D
        y_test_np = y_test_np.flatten()
        test_predictions_np = test_predictions_np.flatten()
        
        # Ensure lengths match
        if len(y_test_np) < len(test_predictions_np):
            test_predictions_np = test_predictions_np[:len(y_test_np)]
        
        test_mse = np.mean((y_test_np - test_predictions_np) ** 2)
    
    # Print test results
    print(f"TEST SET R² SCORE: {test_score:.4f}")
    print(f"TEST SET MSE: {test_mse:.4f}")
    
    # # Plot the results - bar chart of CV scores
    # plt.subplot(3, 3, i*3+1)
    # plt.bar(range(1, k_folds+1), cv_scores, color='skyblue')
    # plt.axhline(y=mean_cv_score, color='red', linestyle='--', label=f'Mean CV: {mean_cv_score:.4f}')
    # plt.axhline(y=test_score, color='green', linestyle='--', label=f'Test: {test_score:.4f}')
    # plt.title(f'{targetCol}: CV vs Test R² Scores')
    # plt.xlabel('CV Fold')
    # plt.ylabel('R² Score')
    # plt.xticks(range(1, k_folds+1))
    # plt.ylim(0, 1.0)
    # plt.legend()
    
    # # Box plot of the CV scores
    # plt.subplot(3, 3, i*3+2)
    # plt.boxplot(cv_scores)
    # plt.title(f'{targetCol}: Distribution of CV Scores')
    # plt.ylabel('R² Score')
    # plt.axhline(y=mean_cv_score, color='red', linestyle='--', label='Mean CV')
    # plt.axhline(y=test_score, color='green', linestyle='--', label='Test Score')
    # plt.legend()
    # plt.ylim(0, 1.0)
    
    # # Actual vs Predicted plot for test set
    # plt.subplot(3, 3, i*3+3)
    # plt.scatter(y_test, test_predictions, alpha=0.5)
    # plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    # plt.title(f'{targetCol}: Test Set Predictions\nR² = {test_score:.4f}')
    # plt.xlabel('Actual Values')
    # plt.ylabel('Predicted Values')
    
    print("---------------------------------------------------------")

# plt.tight_layout()
# plt.show()




