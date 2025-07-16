import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

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

    # SVM
    svm = SVC(kernel='linear', random_state=42)
    svm.fit(X_train, y_train)
    svm_predictions = svm.predict(X_test)
    svm_accuracy = accuracy_score(y_test, svm_predictions)
    print(f"SVM Accuracy: {svm_accuracy:.4f}")

    # Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    nb_predictions = nb.predict(X_test)
    nb_accuracy = accuracy_score(y_test, nb_predictions)
    print(f"Naive Bayes Accuracy: {nb_accuracy:.4f}")

# Main function
def main():
    try:
        # Load the dataset
        file_path = r'C:\86\projectmaj\data_cleaning_toolkit\cleaned_datas.csv'
        df = pd.read_csv(file_path, encoding='utf-8')
        print("Dataset loaded successfully.")
        print(f"Dataset shape: {df.shape}")

        # Detect the target column
        target_column = detect_target_column(df)
        print(f"Target column detected: {target_column}")

        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        print("Features and target separated.")

        # Check for missing values
        print("Missing values in features:")
        print(X.isnull().sum())

        # Fill missing values (if any)
        X = X.fillna(0)

        # Ensure all features are numeric
        X = X.apply(pd.to_numeric, errors='coerce')

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print("Data split into training and testing sets.")

        # Apply ML algorithms
        apply_ml_algorithms(X_train, X_test, y_train, y_test)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()