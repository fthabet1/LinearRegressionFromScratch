import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def trainTestSplitData(df, targetCol, testSize=0.2):
    """
    Split a dataset into training and testing sets.

    Parameters:
    -----------
    df: DataFrame, the dataset to split
    targetCol: str, name of the target column
    testSize: float, proportion of the dataset to include in the test split

    Returns:
    --------
    X_train: DataFrame, training data
    X_test: DataFrame, testing data
    y_train: Series, training target values
    y_test: Series, testing target values
    """
    X = df.drop(targetCol, axis=1)
    y = df[targetCol]

    # Calculate the number of rows for the training set
    trainRows = int((1 - testSize) * len(df))

    # Create random indices for shuffling
    indices = np.random.permutation(len(df))

    # Split the data using random indices
    X_train, X_test = X.iloc[indices[:trainRows]], X.iloc[indices[trainRows:]]
    y_train, y_test = y.iloc[indices[:trainRows]], y.iloc[indices[trainRows:]]

    return X_train, X_test, y_train, y_test

def kFoldCrossValidation(X, y, nFolds=5):
    """
    Create the k-folds for cross validation on a dataset.

    Parameters:
    -----------
    X: DataFrame, the dataset to split
    y: Series, target values
    nFolds: int, number of folds

    Returns:
    --------
    splits: list of tuples, (X_train, X_test, y_train, y_test)
    """

    splits = []
    foldSize = len(X) // nFolds

    for i in range(nFolds):
        start = i * foldSize
        end = (i + 1) * foldSize

        X_test = X.iloc[start:end]
        y_test = y.iloc[start:end]

        X_train = pd.concat([X.iloc[:start], X.iloc[end:]])
        y_train = pd.concat([y.iloc[:start], y.iloc[end:]])

        splits.append((X_train, X_test, y_train, y_test))

    return splits