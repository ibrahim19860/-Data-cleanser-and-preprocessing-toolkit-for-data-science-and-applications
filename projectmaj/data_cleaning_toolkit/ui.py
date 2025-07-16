# import pandas as pd
# import tkinter as tk
# from tkinter import filedialog, messagebox
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
# import data_cleaning_toolkit as dct  # Custom cleaning toolkit

# def detect_target_column(df):
#     return 'target' if 'target' in df.columns else df.columns[-1]

# def clean_dataset(df):
#     for column in df.select_dtypes(include=['number']).columns:
#         df[column] = df[column].fillna(df[column].mean())
#     for column in df.select_dtypes(include=['object']).columns:
#         df[column] = df[column].fillna(df[column].mode()[0])
#     for column in df.select_dtypes(include=['number']).columns:
#         df = dct.remove_outliers_iqr(df, column)
#         if df[column].std() != 0:
#             df = dct.z_score_outliers(df, column)
#     for column in df.select_dtypes(include=['object']).columns:
#         df = dct.encode_categorical_one_hot(df, column)
#     return df

# def apply_ml_algorithm(X_train, X_test, y_train, y_test, algorithm):
#     if algorithm == 'Random Forest':
#         model = RandomForestClassifier(n_estimators=100, random_state=42)
#     elif algorithm == 'Logistic Regression':
#         model = LogisticRegression(max_iter=1000, random_state=42)
#     else:
#         messagebox.showerror("Error", "Invalid Algorithm Selected")
#         return None
    
#     model.fit(X_train, y_train)
#     predictions = model.predict(X_test)
#     accuracy = accuracy_score(y_test, predictions)
#     messagebox.showinfo("Accuracy", f"{algorithm} Accuracy: {accuracy:.4f}")

# def load_and_process_file():
#     file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
#     if not file_path:
#         return
    
#     df = pd.read_csv(file_path)
#     df = clean_dataset(df)
#     target_column = detect_target_column(df)
#     X = df.drop(columns=[target_column])
#     y = df[target_column]
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
#     def on_algorithm_selected():
#         selected_algorithm = algorithm_var.get()
#         apply_ml_algorithm(X_train, X_test, y_train, y_test, selected_algorithm)
    
#     algo_window = tk.Toplevel(root)
#     algo_window.title("Select Algorithm")
#     tk.Label(algo_window, text="Choose an algorithm:").pack()
#     algorithm_var = tk.StringVar(value="Random Forest")
#     tk.Radiobutton(algo_window, text="Random Forest", variable=algorithm_var, value="Random Forest").pack()
#     tk.Radiobutton(algo_window, text="Logistic Regression", variable=algorithm_var, value="Logistic Regression").pack()
#     tk.Button(algo_window, text="Run", command=on_algorithm_selected).pack()

# root = tk.Tk()
# root.title("ML Accuracy Checker")
# tk.Button(root, text="Upload CSV", command=load_and_process_file).pack()
# root.mainloop()


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import data_cleaning_toolkit as dct  # Assuming this is your custom cleaning toolkit
import tkinter as tk
from tkinter import filedialog, messagebox

# Function to automatically detect the target column
def detect_target_column(df):
    # Assuming the target column is the last one or named 'target'
    if 'target' in df.columns:
        return 'target'
    else:
        return df.columns[-1]

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

# Function to apply ML algorithms and calculate accuracy
def apply_ml_algorithm(X_train, X_test, y_train, y_test, algorithm):
    if algorithm == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif algorithm == "Logistic Regression":
        model = LogisticRegression(max_iter=1000, random_state=42)
    elif algorithm == "SVM":
        model = SVC(kernel='linear', random_state=42)
    elif algorithm == "KNN":
        model = KNeighborsClassifier(n_neighbors=5)
    else:
        raise ValueError("Invalid algorithm selected.")
    
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy

# Main function to process the dataset and run the selected algorithm
def process_dataset(file_path, algorithm):
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

    # Apply the selected ML algorithm
    accuracy = apply_ml_algorithm(X_train, X_test, y_train, y_test, algorithm)
    return accuracy

# GUI Application
class MLApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Machine Learning App")
        self.root.geometry("400x300")
        
        # File Upload Button
        self.file_path = None
        self.upload_button = tk.Button(root, text="Upload CSV File", command=self.upload_file)
        self.upload_button.pack(pady=20)
        
        # Algorithm Selection Dropdown
        self.algorithm_var = tk.StringVar(value="Random Forest")
        self.algorithm_menu = tk.OptionMenu(root, self.algorithm_var, "Random Forest", "Logistic Regression", "SVM", "KNN")
        self.algorithm_menu.pack(pady=20)
        
        # Run Button
        self.run_button = tk.Button(root, text="Run Algorithm", command=self.run_algorithm)
        self.run_button.pack(pady=20)
        
        # Accuracy Label
        self.accuracy_label = tk.Label(root, text="Accuracy: ")
        self.accuracy_label.pack(pady=20)
    
    def upload_file(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if self.file_path:
            messagebox.showinfo("File Uploaded", f"File {self.file_path} uploaded successfully.")
    
    def run_algorithm(self):
        if not self.file_path:
            messagebox.showerror("Error", "Please upload a CSV file first.")
            return
        
        algorithm = self.algorithm_var.get()
        try:
            accuracy = process_dataset(self.file_path, algorithm)
            self.accuracy_label.config(text=f"Accuracy: {accuracy:.4f}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = MLApp(root)
    root.mainloop()

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score
# import data_cleaning_toolkit as dct  # Assuming this is your custom cleaning toolkit
# import streamlit as st

# # Function to automatically detect the target column
# def detect_target_column(df):
#     # Assuming the target column is the last one or named 'target'
#     if 'target' in df.columns:
#         return 'target'
#     else:
#         return df.columns[-1]

# # Function to clean the dataset
# def clean_dataset(df):
#     # Handle missing values dynamically based on dataset
#     for column in df.select_dtypes(include=['number']).columns:
#         df[column] = df[column].fillna(df[column].mean())
    
#     for column in df.select_dtypes(include=['object']).columns:
#         df[column] = df[column].fillna(df[column].mode()[0])
    
#     # Remove outliers dynamically
#     for column in df.select_dtypes(include=['number']).columns:
#         df = dct.remove_outliers_iqr(df, column)
#         if df[column].std() != 0:  # Prevent division by zero
#             df = dct.z_score_outliers(df, column)
    
#     # Encode categorical data
#     for column in df.select_dtypes(include=['object']).columns:
#         df = dct.encode_categorical_one_hot(df, column)
    
#     return df

# # Function to apply ML algorithms and calculate accuracy
# def apply_ml_algorithm(X_train, X_test, y_train, y_test, algorithm):
#     if algorithm == "Random Forest":
#         model = RandomForestClassifier(n_estimators=100, random_state=42)
#     elif algorithm == "Logistic Regression":
#         model = LogisticRegression(max_iter=1000, random_state=42)
#     elif algorithm == "SVM":
#         model = SVC(kernel='linear', random_state=42)
#     elif algorithm == "KNN":
#         model = KNeighborsClassifier(n_neighbors=5)
#     else:
#         raise ValueError("Invalid algorithm selected.")
    
#     model.fit(X_train, y_train)
#     predictions = model.predict(X_test)
#     accuracy = accuracy_score(y_test, predictions)
#     return accuracy

# # Streamlit App
# def main():
#     st.title("Machine Learning App with Streamlit")
#     st.write("Upload a CSV file, select an algorithm, and view the accuracy.")

#     # File Upload
#     uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
#     if uploaded_file is not None:
#         df = pd.read_csv(uploaded_file)
#         st.write("Original Dataset:")
#         st.write(df.head())

#         # Clean the dataset
#         df = clean_dataset(df)
#         st.write("Cleaned Dataset:")
#         st.write(df.head())

#         # Detect the target column
#         target_column = detect_target_column(df)
#         st.write(f"Target column detected: {target_column}")

#         # Separate features and target
#         X = df.drop(columns=[target_column])
#         y = df[target_column]

#         # Split the data into training and testing sets
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#         st.write(f"Training data shape: {X_train.shape}, Testing data shape: {X_test.shape}")

#         # Algorithm Selection
#         algorithm = st.selectbox("Select an algorithm", ["Random Forest", "Logistic Regression", "SVM", "KNN"])

#         # Run the selected algorithm
#         if st.button("Run Algorithm"):
#             accuracy = apply_ml_algorithm(X_train, X_test, y_train, y_test, algorithm)
#             st.success(f"Accuracy of {algorithm}: {accuracy:.4f}")

# # Run the Streamlit app
# if __name__ == "__main__":
#     main()