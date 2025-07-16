import pandas as pd
import numpy as np

def remove_outliers_iqr(dataframe, column_name):
    """Remove outliers from a dataframe using the IQR method for a specified column.
    
    Args:
        dataframe (pd.DataFrame): DataFrame containing the data.
        column_name (str): Name of the column to check for outliers.
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed.
    """
    Q1 = dataframe[column_name].quantile(0.25)
    Q3 = dataframe[column_name].quantile(0.75)
    IQR = Q3 - Q1
    dataframe_clean = dataframe[~((dataframe[column_name] < (Q1 - 1.5 * IQR)) |(dataframe[column_name] > (Q3 + 1.5 * IQR)))]
    return dataframe_clean

def z_score_outliers(dataframe, column_name, threshold=3):
    """Remove outliers from a dataframe using Z-score method for a specified column.
    
    Args:
        dataframe (pd.DataFrame): DataFrame containing the data.
        column_name (str): Name of the column to check for outliers.
        threshold (float): Z-score threshold to identify outliers.
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed.
    """
    mean_val = np.mean(dataframe[column_name])
    std_val = np.std(dataframe[column_name])
    z_scores = [(y - mean_val) / std_val for y in dataframe[column_name]]
    return dataframe[np.abs(z_scores) < threshold]
