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



















# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score
# import data_cleaning_toolkit as dct  # Custom cleaning toolkit
# import streamlit as st
# import time

# # Set page title and icon
# st.set_page_config(page_title="ML Accuracy Checker", page_icon="🤖")

# # Custom CSS for styling
# st.markdown(
#     """
#     <style>
#     .stButton button {
#         background-color: #4CAF50;
#         color: white;
#         font-size: 16px;
#         padding: 10px 24px;
#         border-radius: 8px;
#         border: none;
#     }
#     .stButton button:hover {
#         background-color: #45a049;
#     }
#     .stSelectbox div {
#         font-size: 16px;
#     }
#     .stFileUploader div {
#         font-size: 16px;
#     }
#     .stMarkdown h1 {
#         color: #4CAF50;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

# # Function to automatically detect the target column
# def detect_target_column(df):
#     return 'target' if 'target' in df.columns else df.columns[-1]

# # Function to clean the dataset
# def clean_dataset(df):
#     # Handle missing values dynamically based on dataset
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
#     st.title("🤖 Machine Learning Accuracy Checker")
#     st.write("Upload a CSV file, select an algorithm, and view the accuracy in real-time!")

#     # File Upload
#     uploaded_file = st.file_uploader("📁 Upload a CSV file", type=["csv"])
#     if uploaded_file is not None:
#         # Read the dataset
#         df = pd.read_csv(uploaded_file)
        
#         # Display original dataset
#         st.subheader("📊 Original Dataset")
#         st.write(df.head())

#         # Clean the dataset
#         with st.spinner("🧹 Cleaning the dataset..."):
#             df = clean_dataset(df)
#             time.sleep(2)  # Simulate cleaning process
        
#         # Display cleaned dataset
#         st.subheader("✨ Cleaned Dataset")
#         st.write(df.head())

#         # Detect the target column
#         target_column = detect_target_column(df)
#         st.write(f"🎯 Target column detected: `{target_column}`")

#         # Separate features and target
#         X = df.drop(columns=[target_column])
#         y = df[target_column]

#         # Split the data into training and testing sets
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#         st.write(f"📈 Training data shape: `{X_train.shape}`, Testing data shape: `{X_test.shape}`")

#         # Algorithm Selection
#         st.subheader("⚙️ Select an Algorithm")
#         algorithm = st.selectbox(
#             "Choose an algorithm:",
#             ["Random Forest", "Logistic Regression", "SVM", "KNN"],
#             index=0,
#         )

#         # Run the selected algorithm
#         if st.button("🚀 Run Algorithm"):
#             with st.spinner(f"🧠 Training {algorithm} model..."):
#                 accuracy = apply_ml_algorithm(X_train, X_test, y_train, y_test, algorithm)
#                 time.sleep(2)  # Simulate training process
            
#             # Display accuracy
#             st.success(f"✅ Accuracy of **{algorithm}**: `{accuracy:.4f}`")

# # Run the Streamlit app
# if __name__ == "__main__":
#     main()































# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score
# import data_cleaning_toolkit as dct  # Custom cleaning toolkit
# import tkinter as tk
# from tkinter import ttk, filedialog, messagebox
# from threading import Thread

# # Function to automatically detect the target column
# def detect_target_column(df):
#     return 'target' if 'target' in df.columns else df.columns[-1]

# # Function to clean the dataset
# def clean_dataset(df):
#     # Handle missing values dynamically based on dataset
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

# # Function to process the dataset in a separate thread
# def process_dataset(file_path, algorithm, callback, progress_callback):
#     def run():
#         try:
#             # Read the dataset in chunks
#             chunksize = 10**5  # Adjust based on your system's memory
#             chunks = pd.read_csv(file_path, chunksize=chunksize)
            
#             # Process and clean the dataset in chunks
#             cleaned_chunks = []
#             total_chunks = sum(1 for _ in pd.read_csv(file_path, chunksize=chunksize))
#             current_chunk = 0
            
#             for chunk in chunks:
#                 cleaned_chunk = clean_dataset(chunk)
#                 cleaned_chunks.append(cleaned_chunk)
#                 current_chunk += 1
#                 progress_callback((current_chunk / total_chunks) * 50)  # 50% for cleaning
            
#             # Combine cleaned chunks into a single DataFrame
#             df = pd.concat(cleaned_chunks, ignore_index=True)
            
#             # Detect the target column
#             target_column = detect_target_column(df)
            
#             # Separate features and target
#             X = df.drop(columns=[target_column])
#             y = df[target_column]

#             # Split the data into training and testing sets
#             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
#             # Apply the selected ML algorithm
#             accuracy = apply_ml_algorithm(X_train, X_test, y_train, y_test, algorithm)
#             progress_callback(100)  # 100% for completion
            
#             # Call the callback function to update the UI
#             callback(f"Accuracy: {accuracy:.4f}")
#         except Exception as e:
#             callback(f"Error: {str(e)}")
    
#     # Start the thread
#     Thread(target=run).start()

# # GUI Application
# class MLApp:
#     def __init__(self, root):
#         self.root = root
#         self.root.title("Machine Learning App")
#         self.root.geometry("500x400")
#         self.root.configure(bg="#f0f0f0")

#         # Custom Fonts
#         self.title_font = ("Helvetica", 16, "bold")
#         self.label_font = ("Helvetica", 12)
#         self.button_font = ("Helvetica", 12, "bold")

#         # Title Label
#         self.title_label = tk.Label(
#             root, text="🤖 Machine Learning Accuracy Checker", font=self.title_font, bg="#f0f0f0"
#         )
#         self.title_label.pack(pady=10)

#         # File Upload Frame
#         self.upload_frame = tk.Frame(root, bg="#f0f0f0")
#         self.upload_frame.pack(pady=10)

#         self.upload_button = tk.Button(
#             self.upload_frame,
#             text="📁 Upload CSV File",
#             font=self.button_font,
#             command=self.upload_file,
#             bg="#4CAF50",
#             fg="white",
#             relief=tk.FLAT,
#         )
#         self.upload_button.pack(side=tk.LEFT, padx=10)

#         self.file_label = tk.Label(
#             self.upload_frame, text="No file selected", font=self.label_font, bg="#f0f0f0"
#         )
#         self.file_label.pack(side=tk.LEFT)

#         # Algorithm Selection Frame
#         self.algorithm_frame = tk.Frame(root, bg="#f0f0f0")
#         self.algorithm_frame.pack(pady=10)

#         self.algorithm_label = tk.Label(
#             self.algorithm_frame, text="Select Algorithm:", font=self.label_font, bg="#f0f0f0"
#         )
#         self.algorithm_label.pack(side=tk.LEFT, padx=10)

#         self.algorithm_var = tk.StringVar(value="Random Forest")
#         self.algorithm_menu = ttk.Combobox(
#             self.algorithm_frame,
#             textvariable=self.algorithm_var,
#             values=["Random Forest", "Logistic Regression", "SVM", "KNN"],
#             font=self.label_font,
#             state="readonly",
#         )
#         self.algorithm_menu.pack(side=tk.LEFT)

#         # Progress Bar
#         self.progress_bar = ttk.Progressbar(root, orient=tk.HORIZONTAL, length=400, mode="determinate")
#         self.progress_bar.pack(pady=20)

#         # Run Button
#         self.run_button = tk.Button(
#             root,
#             text="🚀 Run Algorithm",
#             font=self.button_font,
#             command=self.run_algorithm,
#             bg="#008CBA",
#             fg="white",
#             relief=tk.FLAT,
#         )
#         self.run_button.pack(pady=10)

#         # Accuracy Label
#         self.accuracy_label = tk.Label(
#             root, text="Accuracy: ", font=self.label_font, bg="#f0f0f0"
#         )
#         self.accuracy_label.pack(pady=10)

#         # Error Label
#         self.error_label = tk.Label(root, text="", font=self.label_font, bg="#f0f0f0", fg="red")
#         self.error_label.pack(pady=10)

#     def upload_file(self):
#         self.file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
#         if self.file_path:
#             self.file_label.config(text=f"📄 {self.file_path.split('/')[-1]}")
#             self.error_label.config(text="")

#     def run_algorithm(self):
#         if not self.file_path:
#             self.error_label.config(text="❌ Please upload a CSV file first.")
#             return
        
#         algorithm = self.algorithm_var.get()
        
#         # Disable the run button to prevent multiple clicks
#         self.run_button.config(state=tk.DISABLED)
#         self.progress_bar["value"] = 0
        
#         # Process the dataset in a separate thread
#         process_dataset(
#             self.file_path,
#             algorithm,
#             self.update_accuracy_label,
#             self.update_progress_bar,
#         )
    
#     def update_accuracy_label(self, message):
#         # Update the accuracy label and re-enable the run button
#         self.accuracy_label.config(text=message)
#         self.run_button.config(state=tk.NORMAL)
#         self.error_label.config(text="")
    
#     def update_progress_bar(self, value):
#         self.progress_bar["value"] = value
#         self.root.update_idletasks()

# # Run the application
# if __name__ == "__main__":
#     root = tk.Tk()
#     app = MLApp(root)
#     root.mainloop()