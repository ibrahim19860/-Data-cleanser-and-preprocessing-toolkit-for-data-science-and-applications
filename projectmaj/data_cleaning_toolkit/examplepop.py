# import pandas as pd
# import numpy as np
# import os
# from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# def detect_column_types(df):
#     """Automatically detects numerical and categorical columns."""
#     num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
#     cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
#     return num_cols, cat_cols

# def handle_missing_values(df):
#     """Fills missing values: mean/median for numerical, mode for categorical."""
#     num_cols, cat_cols = detect_column_types(df)
#     for col in num_cols:
#         df[col] = df[col].fillna(df[col].median())
#     for col in cat_cols:
#         df[col] = df[col].fillna(df[col].mode()[0])
#     return df

# def remove_outliers(df, threshold=1.5):
#     """Removes outliers using the IQR method."""
#     num_cols, _ = detect_column_types(df)
#     for col in num_cols:
#         Q1 = df[col].quantile(0.25)
#         Q3 = df[col].quantile(0.75)
#         IQR = Q3 - Q1
#         lower_bound = Q1 - (threshold * IQR)
#         upper_bound = Q3 + (threshold * IQR)
#         df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
#     return df

# def encode_categorical(df):
#     """Encodes categorical variables: One-Hot for small categories, Label Encoding for large."""
#     _, cat_cols = detect_column_types(df)
#     for col in cat_cols:
#         if df[col].nunique() <= 10:  # One-hot encode small categories
#             df = pd.get_dummies(df, columns=[col], drop_first=True)
#         else:  # Label encode large categories
#             le = LabelEncoder()
#             df[col] = le.fit_transform(df[col])
#     return df

# def normalize_numerical(df):
#     """Normalizes numerical columns using Min-Max Scaling."""
#     num_cols, _ = detect_column_types(df)
#     scaler = MinMaxScaler()
#     df[num_cols] = scaler.fit_transform(df[num_cols])
#     return df

# def optimize_memory(df):
#     """Optimizes memory usage by converting data types."""
#     for col in df.select_dtypes(include=['int64']).columns:
#         df[col] = df[col].astype('int32')
#     for col in df.select_dtypes(include=['float64']).columns:
#         df[col] = df[col].astype('float32')
#     return df

# def preprocess_supermarket_data(df):
#     """Custom cleaning for the supermarket dataset."""
#     if 'Invoice ID' in df.columns:
#         df.drop(columns=['Invoice ID'], inplace=True)  # Drop unique identifier
#     df['Date'] = pd.to_datetime(df['Date'], errors='coerce')  # Convert to datetime
#     if 'Time' in df.columns:
#         df['Time'] = df['Time'].astype(str).str.strip()  # Ensure Time is treated as string and cleaned
#         df['Hour'] = pd.to_datetime(df['Time'], format='%H:%M', errors='coerce').dt.hour  # Extract hour from time
#         df.drop(columns=['Time'], inplace=True)  # Remove original time column
#     df = handle_missing_values(df)
#     df = remove_outliers(df)
#     df = encode_categorical(df)
#     df = normalize_numerical(df)
#     df = optimize_memory(df)
#     return df

# def convert_csv_to_excel(csv_file, excel_file):
#     """Converts a CSV file to an Excel file."""
#     df = pd.read_csv(csv_file)
#     df.to_excel(excel_file, index=False)
#     print(f"CSV successfully converted to Excel: {excel_file}")

# if __name__ == "__main__":
#     # Define file paths
#     csv_path = r"C:\86\projectmaj\supermarket_sales - Sheet1.csv"
#     excel_path = r"C:\86\projectmaj\supermarket_sales - Sheet1.xlsx"
    
#     # Check if CSV exists, otherwise read Excel
#     if os.path.exists(csv_path):
#         df = pd.read_csv(csv_path)
#     elif os.path.exists(excel_path):
#         df = pd.read_excel(excel_path)
#     else:
#         print("Error: No valid file found. Check the file name and location.")
#         exit()
    
#     print("Original DataFrame:")
#     pd.set_option('display.max_rows', None)  # Show all rows
#     pd.set_option('display.max_columns', None)  # Show all columns
#     print(df)
    
#     df_cleaned = preprocess_supermarket_data(df)
#     print("\nCleaned DataFrame:")
#     print(df_cleaned)
    
#     # Save cleaned dataset
#     cleaned_csv_path = r"C:\86\projectmaj\cleaned_supermarket_data.csv"
#     df_cleaned.to_csv(cleaned_csv_path, index=False)
#     print(f"Cleaned CSV file saved successfully: {cleaned_csv_path}")
    
#     # Convert to Excel
#     cleaned_excel_path = r"C:\86\projectmaj\cleaned_supermarket_data.xlsx"
#     convert_csv_to_excel(cleaned_csv_path, cleaned_excel_path)


# # with all estimators ibbu
# import pandas as pd
# import numpy as np
# import os
# from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression, LinearRegression
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

# # Step 1: Data Preprocessing Functions

# def detect_column_types(df):
#     """Automatically detects numerical and categorical columns."""
#     num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
#     cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
#     return num_cols, cat_cols

# def handle_missing_values(df):
#     """Fills missing values: mean/median for numerical, mode for categorical."""
#     num_cols, cat_cols = detect_column_types(df)
#     for col in num_cols:
#         df[col] = df[col].fillna(df[col].median())
#     for col in cat_cols:
#         df[col] = df[col].fillna(df[col].mode()[0])
#     return df

# def remove_outliers(df, threshold=1.5):
#     """Removes outliers using the IQR method."""
#     num_cols, _ = detect_column_types(df)
#     for col in num_cols:
#         Q1 = df[col].quantile(0.25)
#         Q3 = df[col].quantile(0.75)
#         IQR = Q3 - Q1
#         lower_bound = Q1 - (threshold * IQR)
#         upper_bound = Q3 + (threshold * IQR)
#         df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
#     return df

# def encode_categorical(df):
#     """Encodes categorical variables: One-Hot for small categories, Label Encoding for large."""
#     _, cat_cols = detect_column_types(df)
#     for col in cat_cols:
#         if df[col].nunique() <= 10:  # One-hot encode small categories
#             df = pd.get_dummies(df, columns=[col], drop_first=True)
#         else:  # Label encode large categories
#             le = LabelEncoder()
#             df[col] = le.fit_transform(df[col])
#     return df

# def normalize_numerical(df):
#     """Normalizes numerical columns using Min-Max Scaling."""
#     num_cols, _ = detect_column_types(df)
#     scaler = MinMaxScaler()
#     df[num_cols] = scaler.fit_transform(df[num_cols])
#     return df

# def optimize_memory(df):
#     """Optimizes memory usage by converting data types."""
#     for col in df.select_dtypes(include=['int64']).columns:
#         df[col] = df[col].astype('int32')
#     for col in df.select_dtypes(include=['float64']).columns:
#         df[col] = df[col].astype('float32')
#     return df

# def preprocess_supermarket_data(df):
#     """Custom cleaning for the supermarket dataset."""
#     if 'Invoice ID' in df.columns:
#         df.drop(columns=['Invoice ID'], inplace=True)  # Drop unique identifier
#     df['Date'] = pd.to_datetime(df['Date'], errors='coerce')  # Convert to datetime
#     if 'Time' in df.columns:
#         df['Time'] = df['Time'].astype(str).str.strip()  # Ensure Time is treated as string and cleaned
#         df['Hour'] = pd.to_datetime(df['Time'], format='%H:%M', errors='coerce').dt.hour  # Extract hour from time
#         df.drop(columns=['Time'], inplace=True)  # Remove original time column
#     df = handle_missing_values(df)
#     df = remove_outliers(df)
#     df = encode_categorical(df)
#     df = normalize_numerical(df)
#     df = optimize_memory(df)
#     return df

# # Step 2: Detect if the target column is for classification or regression
# def detect_problem_type(y):
#     """Detects if the target column is for classification or regression."""
#     unique_values = y.nunique()
#     if unique_values <= 10:  # Arbitrary threshold for classification
#         return "classification"
#     else:
#         return "regression"

# # Step 3: Train and evaluate models based on problem type
# def train_and_evaluate_models(X_train, X_test, y_train, y_test, problem_type):
#     if problem_type == "classification":
#         models = {
#             "Logistic Regression": LogisticRegression(),
#             "Random Forest Classifier": RandomForestClassifier(random_state=42)
#         }
#     else:
#         models = {
#             "Linear Regression": LinearRegression(),
#             "Random Forest Regressor": RandomForestRegressor(random_state=42)
#         }

#     results = {}
#     for model_name, model in models.items():
#         model.fit(X_train, y_train)
#         y_pred = model.predict(X_test)
        
#         if problem_type == "classification":
#             accuracy = accuracy_score(y_test, y_pred)
#             results[model_name] = accuracy
#             print(f"{model_name} Accuracy: {accuracy:.2f}")
#         else:
#             mse = mean_squared_error(y_test, y_pred)
#             r2 = r2_score(y_test, y_pred)
#             results[model_name] = {'MSE': mse, 'R2': r2}
#             print(f"{model_name} - MSE: {mse:.2f}, R2: {r2:.2f}")

#     return results

# # Step 4: Load and preprocess data
# csv_path = r"C:\86\projectmaj\supermarket_sales - Sheet1.csv"
# excel_path = r"C:\86\projectmaj\supermarket_sales - Sheet1.xlsx"

# if os.path.exists(csv_path):
#     df = pd.read_csv(csv_path)
# elif os.path.exists(excel_path):
#     df = pd.read_excel(excel_path)
# else:
#     print("Error: No valid file found. Check the file name and location.")
#     exit()

# df_cleaned = preprocess_supermarket_data(df)

# # Step 5: Define target column
# target_column = "Total"  # Replace with your target column name

# if target_column not in df_cleaned.columns:
#     raise ValueError(f"Target column '{target_column}' not found in the dataset.")

# # Step 6: Split data into features (X) and target (y)
# X = df_cleaned.drop(columns=[target_column])
# y = df_cleaned[target_column]

# # Step 7: Detect problem type
# problem_type = detect_problem_type(y)
# print(f"Problem Type: {problem_type}")

# # Step 8: Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Step 9: Standardize numerical features (excluding datetime columns)
# # Identify datetime columns
# datetime_cols = df_cleaned.select_dtypes(include=['datetime64']).columns.tolist()

# # Exclude datetime columns from scaling
# numerical_cols = [col for col in X.columns if col not in datetime_cols]

# # Standardize only numerical columns
# scaler = StandardScaler()
# X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
# X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# # Step 10: Train and evaluate models
# results = train_and_evaluate_models(X_train, X_test, y_train, y_test, problem_type)

# # Step 11: Print the best model
# if problem_type == "classification":
#     best_model = max(results, key=results.get)
#     print(f"\nBest Model: {best_model} with Accuracy: {results[best_model]:.2f}")
# else:
#     best_model = min(results, key=lambda x: results[x]['MSE'])
#     print(f"\nBest Model: {best_model} with MSE: {results[best_model]['MSE']:.2f}")



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
