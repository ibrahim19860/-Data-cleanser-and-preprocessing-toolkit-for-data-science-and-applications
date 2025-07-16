import pandas as pd
import data_cleaning_toolkit as dct
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import inspect
import sklearn
from sklearn.utils import all_estimators

def suggest_target_column(df):
    # Suggest target column based on unique values and correlation
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 1:
        correlation_matrix = df.corr().abs()
        target_suggestion = correlation_matrix.sum().idxmax()
    else:
        target_suggestion = df.columns[-1]  # Default to last column
    
    return target_suggestion

def main():
    file_path = input("Enter the CSV file path: ")
    df = pd.read_csv(file_path)
    
    print("Original DataFrame:")
    print(df.head())
    print(f"Initial shape: {df.shape}")
    
    # Handle missing values dynamically based on dataset
    for column in df.select_dtypes(include=['number']).columns:
        df[column] = df[column].fillna(df[column].mean())
    
    for column in df.select_dtypes(include=['object']).columns:
        df[column] = df[column].fillna(df[column].mode()[0])
    
    print(f"Shape after filling missing values: {df.shape}")
    
    # Remove outliers dynamically
    for column in df.select_dtypes(include=['number']).columns:
        df = dct.remove_outliers_iqr(df, column)
        print(f"Shape after IQR outlier removal ({column}): {df.shape}")
        
        if df[column].std() != 0:  # Prevent division by zero
            df = dct.z_score_outliers(df, column)
        print(f"Shape after Z-score outlier removal ({column}): {df.shape}")
    
    # Encode categorical data
    for column in df.select_dtypes(include=['object']).columns:
        df = dct.encode_categorical_one_hot(df, column)
    
    print(f"Shape after encoding categorical data: {df.shape}")
    print("\nCleaned DataFrame:")
    print(df.head())
    
    # Save cleaned data
    output_file = "cleaned_data.csv"
    df.to_csv(output_file, index=False)
    print(f"Cleaned data saved to {output_file}")
    
    # Suggest a target column
    suggested_target = suggest_target_column(df)
    print(f"Suggested target column: {suggested_target}")
    
    target_column = input(f"Enter the target column for prediction (Press Enter to use suggested: {suggested_target}): ")
    target_column = target_column if target_column else suggested_target
    
    if target_column not in df.columns:
        print("Invalid target column!")
        return
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    classifiers = dict(all_estimators(type_filter='classifier'))
    
    for name, Classifier in classifiers.items():
        try:
            model = Classifier()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"{name} Accuracy: {accuracy:.2f}")
        except Exception as e:
            print(f"Skipping {name} due to error: {e}")

if __name__ == "__main__":
    main()
