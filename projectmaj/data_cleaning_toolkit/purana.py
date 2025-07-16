import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import data_cleaning_toolkit as dct  # Custom cleaning toolkit
import tkinter as tk
from tkinter import filedialog, messagebox
from threading import Thread

# Function to automatically detect the target column
def detect_target_column(df):
    return 'target' if 'target' in df.columns else df.columns[-1]

# Function to clean the dataset
def clean_dataset(df):
    # Handle missing values dynamically based on dataset
    for column in df.select_dtypes(include=['number']).columns:
        df[column] = df[column].fillna(df[column].mean())
    for column in df.select_dtypes(include=['object']).columns:
        df[column] = df[column].fillna(df[column].mode()[0])
    for column in df.select_dtypes(include=['number']).columns:
        df = dct.remove_outliers_iqr(df, column)
        if df[column].std() != 0:
            df = dct.z_score_outliers(df, column)
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

# Function to process the dataset in a separate thread
def process_dataset(file_path, algorithm, callback):
    def run():
        try:
            # Read the dataset in chunks
            chunksize = 10**5  # Adjust based on your system's memory
            chunks = pd.read_csv(file_path, chunksize=chunksize)
            
            # Process and clean the dataset in chunks
            cleaned_chunks = []
            for chunk in chunks:
                cleaned_chunk = clean_dataset(chunk)
                cleaned_chunks.append(cleaned_chunk)
            
            # Combine cleaned chunks into a single DataFrame
            df = pd.concat(cleaned_chunks, ignore_index=True)
            
            # Detect the target column
            target_column = detect_target_column(df)
            
            # Separate features and target
            X = df.drop(columns=[target_column])
            y = df[target_column]

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Apply the selected ML algorithm
            accuracy = apply_ml_algorithm(X_train, X_test, y_train, y_test, algorithm)
            
            # Call the callback function to update the UI
            callback(f"Accuracy: {accuracy:.4f}")
        except Exception as e:
            callback(f"Error: {str(e)}")
    
    # Start the thread
    Thread(target=run).start()

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
        
        # Disable the run button to prevent multiple clicks
        self.run_button.config(state=tk.DISABLED)
        
        # Process the dataset in a separate thread
        process_dataset(self.file_path, algorithm, self.update_accuracy_label)
    
    def update_accuracy_label(self, message):
        # Update the accuracy label and re-enable the run button
        self.accuracy_label.config(text=message)
        self.run_button.config(state=tk.NORMAL)

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = MLApp(root)
    root.mainloop()