import pandas as pd

def fill_missing_with_mean(dataframe, column_name):
    """Fill missing values in specified column using mean.
    
    Args:
        dataframe (pd.DataFrame): DataFrame containing the data.
        column_name (str): Name of the column to fill missing values.
        
    Returns:
        pd.DataFrame: DataFrame with missing values filled.
    """
    try:
        mean_value = dataframe[column_name].mean()
        dataframe[column_name].fillna(mean_value, inplace=True)
    except KeyError:
        print(f"Column {column_name} does not exist in the DataFrame.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return dataframe

def fill_missing_with_median(dataframe, column_name):
    """Fill missing values in specified column using median.
    
    Args:
        dataframe (pd.DataFrame): DataFrame containing the data.
        column_name (str): Name of the column to fill missing values.
        
    Returns:
        pd.DataFrame: DataFrame with missing values filled.
    """
    try:
        median_value = dataframe[column_name].median()
        dataframe[column_name].fillna(median_value, inplace=True)
    except KeyError:
        print(f"Column {column_name} does not exist in the DataFrame.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return dataframe
