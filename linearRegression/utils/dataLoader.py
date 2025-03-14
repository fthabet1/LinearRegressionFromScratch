import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
import os


def loadDatasetFromKaggle(path):
    """
    Load a dataset from Kaggle using the KaggleHub API.

    Parameters:
    -----------
    path: str, path to the dataset on Kaggle

    Returns:
    --------
    df: DataFrame, the dataset as a pandas DataFrame
    """

    path = kagglehub.dataset_download(path)
    files = os.listdir(path)
    csvFile = os.path.join(path, files[0])
    df = pd.read_csv(csvFile)

    return df

def loadDatasetFromCSV(path):
    """
    Load a dataset from a CSV file.

    Parameters:
    -----------
    path: str, path to the CSV file
    targetCol: str, name of the target column
    featureCols: list, names of the feature columns

    Returns:
    --------
    df: DataFrame, the dataset as a pandas DataFrame
    """

    df = pd.read_csv(path)

    return df

def loadDatasetFromExcel(path):
    """
    Load a dataset from an Excel file.

    Parameters:
    -----------
    path: str, path to the Excel file

    Returns:
    --------
    df: DataFrame, the dataset as a pandas DataFrame
    """

    df = pd.read_excel(path)

    return df