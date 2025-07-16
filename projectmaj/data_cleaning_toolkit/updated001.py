import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import streamlit as st
import time

# Set page title and icon
st.set_page_config(page_title="ML Accuracy Checker", page_icon="🤖")

# Custom CSS for styling
st.markdown("""
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
    .stSelectbox div, .stFileUploader div {
        font-size: 16px;
    }
    .stMarkdown h1 {
        color: #4CAF50;
    }
    </style>
    """, unsafe_allow_html=True)

# Function to detect target column
def detect_target_column(df):
    if 'target' in df.columns:
        return 'target'
    else:
        candidate = df.columns[-1]
        if df[candidate].nunique() < len(df) * 0.5:
            return candidate
        else:
            raise ValueError("Cannot detect a valid target column.")

# Clean dataset without external libraries
def clean_dataset(df, target_column):
    df = df.copy()

    # Fill missing values
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            df[col] = df[col].fillna(df[col].mean())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

    # One-hot encode categorical columns (excluding target)
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if target_column in categorical_cols:
        categorical_cols.remove(target_column)
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    return df

# ML model training
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
    return accuracy_score(y_test, predictions)

# Streamlit app
def main():
    st.title("🤖 Machine Learning Accuracy Checker")
    st.write("Upload a CSV file, select an algorithm, and view the accuracy in real-time!")

    uploaded_file = st.file_uploader("📁 Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.subheader("📊 Original Dataset")
            st.write(df.head())

            target_column = detect_target_column(df)
            st.write(f"🎯 Target column detected: `{target_column}`")

            y = df[target_column]
            X = df.drop(columns=[target_column])

            # Combine for uniform cleaning
            combined_df = pd.concat([X, y], axis=1)
            with st.spinner("🧹 Cleaning the dataset..."):
                cleaned_df = clean_dataset(combined_df, target_column)
                time.sleep(1)

            st.subheader("✨ Cleaned Dataset")
            st.write(cleaned_df.head())

            X = cleaned_df.drop(columns=[target_column])
            y = cleaned_df[target_column]

            if y.dtype == 'object':
                y = pd.factorize(y)[0]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            st.write(f"📈 Training data shape: {X_train.shape}, Testing data shape: {X_test.shape}")

            st.subheader("⚙ Select an Algorithm")
            algorithm = st.selectbox("Choose an algorithm:", ["Random Forest", "Logistic Regression", "SVM", "KNN"])

            if st.button("🚀 Run Algorithm"):
                with st.spinner(f"🧠 Training {algorithm} model..."):
                    accuracy = apply_ml_algorithm(X_train, X_test, y_train, y_test, algorithm)
                    time.sleep(1)
                st.success(f"✅ Accuracy of *{algorithm}*: {accuracy:.4f}")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# Correct entry point
if __name__ == "__main__":
    main()
