# import pandas as pd
# import data_cleaning_toolkit as dct

# def main():
#     data = {
#         'Age': [25, 22, None, 28, 35, 29, None, 40],
#         'Salary': [50000, 48000, 51000, None, 55000, None, 52000, 56],
#         'Height': [5.5, 5.42, 5.75, 5.58, None, 5.92, 5.4, 5.8],
#         'Gender': ['Male','Female','Female', 'Male', 'Male', 'Male', 'Female', 'Male']
#     }
#     df = pd.DataFrame(data)
#     print("Original DataFrame:")
#     print(df)

#     df = dct.fill_missing_with_mean(df, 'Age')
#     df = dct.fill_missing_with_mean(df, 'Salary')
#     df = dct.fill_missing_with_median(df, 'Height')

#     df = dct.remove_outliers_iqr(df, 'Salary')
#     df = dct.z_score_outliers(df, 'Height')

#     # df = dct.min_max_scaling(df, 'Age')
#     df = dct.encode_categorical_one_hot(df, 'Gender')
#     # df = dct.encode_categorical_label(df, 'Gender')

#     print("\nCleaned DataFrame:")
#     print(df)

# if __name__ == "__main__":
#     main()





















#plain old one below





# import pandas as pd
# import data_cleaning_toolkit as dct

# def main():
#     file_path = input("Enter the CSV file path: ")
#     df = pd.read_csv(file_path)
    
#     print("Original DataFrame:")
#     print(df.head())
#     print(f"Initial shape: {df.shape}")
    
#     # Handle missing values dynamically based on dataset
#     for column in df.select_dtypes(include=['number']).columns:
#         df[column] = df[column].fillna(df[column].mean())
    
#     for column in df.select_dtypes(include=['object']).columns:
#         df[column] = df[column].fillna(df[column].mode()[0])
    
#     print(f"Shape after filling missing values: {df.shape}")
    
#     # Remove outliers dynamically
#     for column in df.select_dtypes(include=['number']).columns:
#         df = dct.remove_outliers_iqr(df, column)
#         print(f"Shape after IQR outlier removal ({column}): {df.shape}")
        
#         if df[column].std() != 0:  # Prevent division by zero
#             df = dct.z_score_outliers(df, column)
#         print(f"Shape after Z-score outlier removal ({column}): {df.shape}")
    
#     # Encode categorical data
#     for column in df.select_dtypes(include=['object']).columns:
#         df = dct.encode_categorical_one_hot(df, column)
    
#     print(f"Shape after encoding categorical data: {df.shape}")
#     print("\nCleaned DataFrame:")
#     print(df.head())
    
#     # Save cleaned data
#     output_file = "cleaned_datas.csv"
#     df.to_csv(output_file, index=False)
#     print(f"Cleaned data saved to {output_file}")

# if __name__ == "__main__":
#     main()

# # #ye main re
# import pandas as pd
# import data_cleaning_toolkit as dct
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score

# def main():
#     file_path = input("Enter the CSV file path: ")
#     df = pd.read_csv(file_path)
    
#     print("Original DataFrame:")
#     print(df.head())
#     print(f"Initial shape: {df.shape}")
    
#     # Handle missing values dynamically based on dataset
#     for column in df.select_dtypes(include=['number']).columns:
#         df[column] = df[column].fillna(df[column].mean())
    
#     for column in df.select_dtypes(include=['object']).columns:
#         df[column] = df[column].fillna(df[column].mode()[0])
    
#     print(f"Shape after filling missing values: {df.shape}")
    
#     # Remove outliers dynamically
#     for column in df.select_dtypes(include=['number']).columns:
#         df = dct.remove_outliers_iqr(df, column)
#         print(f"Shape after IQR outlier removal ({column}): {df.shape}")
        
#         if df[column].std() != 0:  # Prevent division by zero
#             df = dct.z_score_outliers(df, column)
#         print(f"Shape after Z-score outlier removal ({column}): {df.shape}")
    
#     # Encode categorical data
#     for column in df.select_dtypes(include=['object']).columns:
#         df = dct.encode_categorical_one_hot(df, column)
    
#     print(f"Shape after encoding categorical data: {df.shape}")
#     print("\nCleaned DataFrame:")
#     print(df.head())
    
#     # Save cleaned data
#     output_file = "cleaned_data.csv"
#     df.to_csv(output_file, index=False)
#     print(f"Cleaned data saved to {output_file}")
    
#     # Machine Learning Model Training
#     target_column = input("Enter the target column for prediction: ")
#     if target_column not in df.columns:
#         print("Invalid target column!")
#         return
    
#     X = df.drop(columns=[target_column])
#     y = df[target_column]
    
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
#     model = RandomForestClassifier()
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
    
#     accuracy = accuracy_score(y_test, y_pred)
#     print(f"Model Accuracy: {accuracy:.2f}")

# if __name__ == "__main__":
#     main()

# all estimators iibu
import pandas as pd
import data_cleaning_toolkit as dct
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import inspect
import sklearn
from sklearn.utils import all_estimators

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
    output_file = "cleaned_data1.csv"
    df.to_csv(output_file, index=False)
    print(f"Cleaned data saved to {output_file}")
    
    # Machine Learning Model Training
    target_column = input("Enter the target column for prediction: ")
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






















# import pandas as pd
# import data_cleaning_toolkit as dct
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score

# def main():
#     file_path = input("Enter the CSV file path: ")
#     df = pd.read_csv(file_path)
    
#     print("Original DataFrame:")
#     print(df.head())
#     print(f"Initial shape: {df.shape}")
    
#     # Handle missing values dynamically based on dataset
#     for column in df.select_dtypes(include=['number']).columns:
#         df[column] = df[column].fillna(df[column].mean())
    
#     for column in df.select_dtypes(include=['object']).columns:
#         df[column] = df[column].fillna(df[column].mode()[0])
    
#     print(f"Shape after filling missing values: {df.shape}")
    
#     # Remove outliers dynamically
#     for column in df.select_dtypes(include=['number']).columns:
#         df = dct.remove_outliers_iqr(df, column)
#         print(f"Shape after IQR outlier removal ({column}): {df.shape}")
        
#         if df[column].std() != 0:  # Prevent division by zero
#             df = dct.z_score_outliers(df, column)
#         print(f"Shape after Z-score outlier removal ({column}): {df.shape}")
    
#     # Encode categorical data
#     for column in df.select_dtypes(include=['object']).columns:
#         df = dct.encode_categorical_one_hot(df, column)
    
#     print(f"Shape after encoding categorical data: {df.shape}")
#     print("\nCleaned DataFrame:")
#     print(df.head())
    
#     # Save cleaned data
#     output_file = "cleaned_data.csv"
#     df.to_csv(output_file, index=False)
#     print(f"Cleaned data saved to {output_file}")
    
#     # Machine Learning Model Training
#     target_column = input("Enter the target column for prediction: ")
#     if target_column not in df.columns:
#         print("Invalid target column!")
#         return
    
#     X = df.drop(columns=[target_column])
#     y = df[target_column]
    
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
#     models = {
#         "Random Forest": RandomForestClassifier(),
#         "Logistic Regression": LogisticRegression(max_iter=1000),
#         "SVM": SVC()
#     }
    
#     for model_name, model in models.items():
#         model.fit(X_train, y_train)
#         y_pred = model.predict(X_test)
#         accuracy = accuracy_score(y_test, y_pred)
#         print(f"{model_name} Accuracy: {accuracy:.2f}")

# if __name__ == "__main__":
#     main()
