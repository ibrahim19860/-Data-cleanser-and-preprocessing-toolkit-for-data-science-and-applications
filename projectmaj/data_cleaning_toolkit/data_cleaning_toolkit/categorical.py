import pandas as pd

def encode_categorical_one_hot(dataframe, column_name):
    """Apply one-hot encoding to a specified categorical column.
    
    Args:
        dataframe (pd.DataFrame): DataFrame containing the data.
        column_name (str): Name of the categorical column to encode.
    
    Returns:
        pd.DataFrame: DataFrame with one-hot encoded column.
    """
    dummies = pd.get_dummies(dataframe[column_name], prefix=column_name)
    dataframe = pd.concat([dataframe, dummies], axis=1)
    dataframe.drop(column_name, axis=1, inplace=True)
    return dataframe

def encode_categorical_label(dataframe, column_name):
    """Apply label encoding to a specified categorical column.
    
    Args:
        dataframe (pd.DataFrame): DataFrame containing the data.
        column_name (str): Name of the categorical column to encode.
    
    Returns:
        pd.DataFrame: DataFrame with label encoded column.
    """
    dataframe[column_name] = dataframe[column_name].astype('category').cat.codes
    return dataframe
