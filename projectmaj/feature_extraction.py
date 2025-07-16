import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures

class FeatureExtraction:
    def __init__(self, df, target_column):
        """
        Initializes the FeatureExtraction class.
        :param df: DataFrame containing the dataset.
        :param target_column: The column name of the target variable.
        """
        self.df = df
        self.target_column = target_column
        self.X = df.drop(columns=[target_column])
        self.y = df[target_column]

    def feature_selection(self):
        """
        Removes low variance features, selects the best features, and applies Recursive Feature Elimination (RFE).
        :return: DataFrame with selected features.
        """
        # Remove low variance features
        selector = VarianceThreshold(threshold=0.01)
        self.X = selector.fit_transform(self.X)

        # Select K best features based on statistical tests
        best_features = SelectKBest(score_func=f_classif, k=min(5, self.X.shape[1]))  # Prevents errors with small datasets
        self.X = best_features.fit_transform(self.X, self.y)

        # Apply Recursive Feature Elimination (RFE) with Random Forest
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        rfe = RFE(model, n_features_to_select=min(5, self.X.shape[1]))  # Selects 5 features or fewer
        self.X = rfe.fit_transform(self.X, self.y)

        return self.X

    def reduce_dimensions(self, n_components=3):
        """
        Applies Principal Component Analysis (PCA) for dimensionality reduction.
        :param n_components: Number of principal components to keep.
        :return: DataFrame with reduced dimensions.
        """
        if self.X.shape[1] > n_components:  # Ensures PCA is applicable
            pca = PCA(n_components=n_components)
            self.X = pca.fit_transform(self.X)
        return self.X

    def create_features(self):
        """
        Generates new features using polynomial transformations, log transformations, and squared features.
        :return: Updated DataFrame with new features.
        """
        df = self.df.copy()

        # Log transformation for positive numerical columns
        for col in df.select_dtypes(include=['number']).columns:
            if df[col].min() > 0:  # Prevents log of zero or negative numbers
                df[f'{col}_log'] = np.log1p(df[col])

        # Squaring numeric features
        for col in df.select_dtypes(include=['number']).columns:
            df[f'{col}_squared'] = df[col] ** 2

        # Polynomial Features (Interaction Terms)
        poly = PolynomialFeatures(degree=2, interaction_only=True)
        poly_features = poly.fit_transform(df.select_dtypes(include=['number']))
        df_poly = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(df.select_dtypes(include=['number']).columns))
        
        # Merge polynomial features
        df = pd.concat([df, df_poly], axis=1)
        return df

    def extract_features(self):
        """
        Runs all feature extraction methods and returns the transformed dataset.
        :return: Transformed feature matrix (X) and target variable (y).
        """
        self.X = self.feature_selection()
        self.X = self.reduce_dimensions()
        self.df = self.create_features()
        return self.df, self.X, self.y
