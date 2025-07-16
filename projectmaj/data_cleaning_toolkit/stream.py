import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import data_cleaning_toolkit as dct  # Custom cleaning toolkit
#from data_cleaning_toolkit.feature_extraction import FeatureExtraction
import streamlit as st
import time

# Set page title and icon
st.set_page_config(page_title="ML Accuracy Checker", page_icon="🤖")

# Custom CSS for styling
st.markdown(
    """
    <style>
    .stButton button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        padding: 10px 24px;
        border-radius: 8px;
        border: none;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    .stSelectbox div {
        font-size: 16px;
    }
    .stFileUploader div {
        font-size: 16px;
    }
    .stMarkdown h1 {
        color: #4CAF50;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

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

# Streamlit App
def main():
    st.title("🤖 Machine Learning Accuracy Checker")
    st.write("Upload a CSV file, select an algorithm, and view the accuracy in real-time!")

    # File Upload
    uploaded_file = st.file_uploader("📁 Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        # Read the dataset
        df = pd.read_csv(uploaded_file)
        
        # Display original dataset
        st.subheader("📊 Original Dataset")
        st.write(df.head())

        # Clean the dataset
        with st.spinner("🧹 Cleaning the dataset..."):
            df = clean_dataset(df)
            time.sleep(2)  # Simulate cleaning process
        
        # Display cleaned dataset
        st.subheader("✨ Cleaned Dataset")
        st.write(df.head())

        # Detect the target column
        target_column = detect_target_column(df)
        st.write(f"🎯 Target column detected: `{target_column}`")

        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        st.write(f"📈 Training data shape: `{X_train.shape}`, Testing data shape: `{X_test.shape}`")

        # Algorithm Selection
        st.subheader("⚙️ Select an Algorithm")
        algorithm = st.selectbox(
            "Choose an algorithm:",
            ["Random Forest", "Logistic Regression", "SVM", "KNN"],
            index=0,
        )

        # Run the selected algorithm
        if st.button("🚀 Run Algorithm"):
            with st.spinner(f"🧠 Training {algorithm} model..."):
                accuracy = apply_ml_algorithm(X_train, X_test, y_train, y_test, algorithm)
                time.sleep(2)  # Simulate training process
            
            # Display accuracy
            st.success(f"✅ Accuracy of **{algorithm}**: `{accuracy:.4f}`")

# Run the Streamlit app
if __name__ == "__main__":
    main()



























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
#     st.write("Upload a CSV file, select the target column, choose an algorithm, and view the accuracy in real-time!")

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

#         # Select target column
#         st.subheader("🎯 Select Target Column")
#         target_column = st.selectbox(
#             "Choose the target column:",
#             df.columns,
#             index=len(df.columns) - 1,  # Default to the last column
#         )

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

# # Function to suggest the best target column
# def suggest_target_column(df):
#     # Use correlation to suggest the best target column
#     numeric_df = df.select_dtypes(include=['number'])
#     if numeric_df.empty:
#         return df.columns[-1]  # Default to the last column if no numeric columns
    
#     # Find the column with the highest average correlation
#     correlation_matrix = numeric_df.corr().abs()
#     target_column = correlation_matrix.mean().idxmax()
#     return target_column

# # Function to suggest the most suitable algorithm
# def suggest_algorithm(df, target_column):
#     # Analyze dataset characteristics
#     num_samples, num_features = df.shape
#     target_type = df[target_column].dtype

#     # Rule-based suggestions
#     if num_samples < 1000:
#         if target_type == 'object' or df[target_column].nunique() <= 2:
#             return "Logistic Regression"  # Small dataset, binary classification
#         else:
#             return "KNN"  # Small dataset, multi-class classification
#     elif num_features > 20:
#         return "Random Forest"  # High-dimensional dataset
#     else:
#         return "SVM"  # Medium-sized dataset with clear separation

# # Streamlit App
# def main():
#     st.title("🤖 Machine Learning Accuracy Checker")
#     st.write("Upload a CSV file, and let the app suggest the best target column and algorithm!")

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

#         # Suggest the best target column
#         target_column = suggest_target_column(df)
#         st.subheader("🎯 Suggested Target Column")
#         st.write(f"The app suggests using `{target_column}` as the target column.")

#         # Suggest the most suitable algorithm
#         suggested_algorithm = suggest_algorithm(df, target_column)
#         st.subheader("⚙️ Suggested Algorithm")
#         st.write(f"The app suggests using **{suggested_algorithm}** for this dataset.")

#         # Separate features and target
#         X = df.drop(columns=[target_column])
#         y = df[target_column]

#         # Split the data into training and testing sets
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#         st.write(f"📈 Training data shape: `{X_train.shape}`, Testing data shape: `{X_test.shape}`")

#         # Run the suggested algorithm
#         if st.button("🚀 Run Suggested Algorithm"):
#             with st.spinner(f"🧠 Training {suggested_algorithm} model..."):
#                 accuracy = apply_ml_algorithm(X_train, X_test, y_train, y_test, suggested_algorithm)
#                 time.sleep(2)  # Simulate training process
            
#             # Display accuracy
#             st.success(f"✅ Accuracy of **{suggested_algorithm}**: `{accuracy:.4f}`")

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
# from data_cleaning_toolkit.feature_extraction import FeatureExtraction
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
#     .stSelectbox div, .stFileUploader div {
#         font-size: 16px;
#     }
#     .stMarkdown h1 {
#         color: #4CAF50;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

# # Function to automatically detect or let the user select the target column
# def detect_target_column(df):
#     default_target = 'target' if 'target' in df.columns else df.columns[-1]
#     return st.selectbox("🎯 Select the Target Column", df.columns, index=df.columns.get_loc(default_target))

# # Function to clean the dataset
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

# # Function to apply ML algorithms and calculate accuracy
# def apply_ml_algorithm(X_train, X_test, y_train, y_test, algorithm):
#     model_map = {
#         "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
#         "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
#         "SVM": SVC(kernel='linear', random_state=42),
#         "KNN": KNeighborsClassifier(n_neighbors=5)
#     }
#     model = model_map.get(algorithm)
#     if model:
#         model.fit(X_train, y_train)
#         predictions = model.predict(X_test)
#         return accuracy_score(y_test, predictions)
#     return None

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
#             time.sleep(1.5)  # Simulate cleaning process
        
#         # Display cleaned dataset
#         st.subheader("✨ Cleaned Dataset")
#         st.write(df.head())

#         # Detect or let the user select the target column
#         target_column = detect_target_column(df)
#         st.write(f"🎯 Target column selected: `{target_column}`")

#         # Feature Extraction Option
#         apply_feature_extraction = st.checkbox("🛠️ Apply Feature Extraction")
        
#         if apply_feature_extraction:
#             with st.spinner("🔍 Extracting features..."):
#                 fe = FeatureExtraction(df, target_column)
#                 df, _, _ = fe.extract_features()
#                 time.sleep(1.5)  # Simulate extraction process
#             st.success("✅ Feature extraction applied!")

#         # Separate features and target
#         X = df.drop(columns=[target_column])
#         y = df[target_column]

#         # Split the data into training and testing sets
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#         st.write(f"📈 Training data shape: `{X_train.shape}`, Testing data shape: `{X_test.shape}`")

#         # Algorithm Selection
#         st.subheader("⚙️ Select an Algorithm")
#         algorithm = st.selectbox("Choose an algorithm:", ["Random Forest", "Logistic Regression", "SVM", "KNN"])

#         # Run the selected algorithm
#         if st.button("🚀 Run Algorithm"):
#             with st.spinner(f"🧠 Training {algorithm} model..."):
#                 accuracy = apply_ml_algorithm(X_train, X_test, y_train, y_test, algorithm)
#                 time.sleep(1.5)  # Simulate training process
            
#             # Display accuracy
#             if accuracy is not None:
#                 st.success(f"✅ Accuracy of **{algorithm}**: `{accuracy:.4f}`")
#             else:
#                 st.error("⚠️ Error: Algorithm could not be trained.")

# # Run the Streamlit app
# if __name__ == "__main__":
#     main()
