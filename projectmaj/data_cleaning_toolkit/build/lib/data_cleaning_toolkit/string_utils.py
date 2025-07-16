def strip_whitespace(dataframe, column_name):
    """Strip leading and trailing whitespace from a specified string column.
    
    Args:
        dataframe (pd.DataFrame): DataFrame containing the data.
        column_name (str): Name of the string column to clean.
    
    Returns:
        pd.DataFrame: DataFrame with cleaned column.
    """
    dataframe[column_name] = dataframe[column_name].str.strip()
    return dataframe

def replace_special_chars(dataframe, column_name, char_to_replace, replacement_char):
    """Replace special characters in a specified string column.
    
    Args:
        dataframe (pd.DataFrame): DataFrame containing the data.
        column_name (str): Name of the string column to clean.
        char_to_replace (str): Character to replace.
        replacement_char (str): Character to replace with.
    
    Returns:
        pd.DataFrame: DataFrame with modified column.
    """
    dataframe[column_name] = dataframe[column_name].str.replace(char_to_replace, replacement_char, regex=True)
    return dataframe
