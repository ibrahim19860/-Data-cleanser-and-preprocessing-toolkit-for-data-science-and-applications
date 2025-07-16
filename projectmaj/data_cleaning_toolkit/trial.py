import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import data_cleaning_toolkit as dct  # Assuming this is your custom cleaning toolkit

# Function to automatically detect the target column
def detect_target_column(df):
    # Assuming the target column is the last one or named 'target'
    if 'target' in df.columns:
        return 'target'
    else:
        return df.columns[-1]

# Function to apply ML algorithms and calculate accuracy
def apply_ml_algorithms(X_train, X_test, y_train, y_test):
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_predictions = rf.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_predictions)
    print(f"Random Forest Accuracy: {rf_accuracy:.4f}")

    #Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    lr_predictions = lr.predict(X_test)
    lr_accuracy = accuracy_score(y_test, lr_predictions)
    print(f"Logistic Regression Accuracy: {lr_accuracy:.4f}")

# Function to clean the dataset
def clean_dataset(df):
    # Handle missing values dynamically based on dataset
    for column in df.select_dtypes(include=['number']).columns:
        df[column] = df[column].fillna(df[column].mean())
    
    for column in df.select_dtypes(include=['object']).columns:
        df[column] = df[column].fillna(df[column].mode()[0])
    
    # Remove outliers dynamically
    for column in df.select_dtypes(include=['number']).columns:
        df = dct.remove_outliers_iqr(df, column)
        if df[column].std() != 0:  # Prevent division by zero
            df = dct.z_score_outliers(df, column)
    
    # Encode categorical data
    for column in df.select_dtypes(include=['object']).columns:
        df = dct.encode_categorical_one_hot(df, column)
    
    return df

# Main function
def main():
    file_path = input("Enter the CSV file path: ")
    chunksize = 10**6  # Adjust based on your system's memory
    chunks = pd.read_csv(file_path, chunksize=chunksize)
    
    # Process and clean the dataset in chunks
    cleaned_chunks = []
    for chunk in chunks:
        print(f"Processing chunk of shape: {chunk.shape}")
        cleaned_chunk = clean_dataset(chunk)
        cleaned_chunks.append(cleaned_chunk)
    
    # Combine cleaned chunks into a single DataFrame
    df = pd.concat(cleaned_chunks, ignore_index=True)
    print(f"Final cleaned dataset shape: {df.shape}")
    
    # Detect the target column
    target_column = detect_target_column(df)
    print(f"Target column detected: {target_column}")

    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training data shape: {X_train.shape}, Testing data shape: {X_test.shape}")

    # Apply ML algorithms
    apply_ml_algorithms(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()