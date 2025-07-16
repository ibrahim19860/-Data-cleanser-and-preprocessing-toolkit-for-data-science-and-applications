import pandas as pd

def min_max_scaling(dataframe, column_name):
    """Apply Min-Max scaling to a specified column.
    
    Args:
        dataframe (pd.DataFrame): DataFrame containing the data.
        column_name (str): Name of the column to scale.
    
    Returns:
        pd.DataFrame: DataFrame with scaled column.
    """
    min_val = dataframe[column_name].min()
    max_val = dataframe[column_name].max()
    dataframe[column_name] = (dataframe[column_name] - min_val) / (max_val - min_val)
    return dataframe

def z_score_normalization(dataframe, column_name):
    """Apply Z-score normalization to a specified column.
    
    Args:
        dataframe (pd.DataFrame): DataFrame containing the data.
        column_name (str): Name of the column to normalize.
    
    Returns:
        pd.DataFrame: DataFrame with normalized column.
    """
    mean_val = dataframe[column_name].mean()
    std_val = dataframe[column_name].std()
    dataframe[column_name] = (dataframe[column_name] - mean_val) / std_val
    return dataframe