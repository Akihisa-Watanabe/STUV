import random

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

# fix seed
np.random.seed(42)
random.seed(42)


class BasicFeatureExtractor:
    def __init__(self, init_time=1, use_scaler=True, use_pca=False, n_components=None):
        self.init_time = init_time
        self.use_scaler = use_scaler
        self.use_pca = use_pca
        self.n_components = n_components

        self.columns_to_drop = ["sessionIndex", "rep"]
        self.scaler = RobustScaler() if use_scaler else None
        self.pca = PCA(n_components) if use_pca else None

    def fit(self, user_labels, time_labels, X):
        # Filter only the initial time data
        mask = time_labels == self.init_time
        filtered_X = X[mask].copy()

        # Drop unnecessary columns
        for col in self.columns_to_drop:
            if col in filtered_X.columns:
                filtered_X.drop(columns=[col], inplace=True)

        # Balancing the dataset
        min_samples = filtered_X.groupby(user_labels[mask]).size().min()
        balanced_data = []

        for user in user_labels[mask].unique():
            user_data = filtered_X[user_labels[mask] == user]
            balanced_data.append(
                user_data.sample(min_samples, replace=False, random_state=42)
            )

        balanced_X = pd.concat(balanced_data, axis=0)

        # Scale the data if required
        if self.use_scaler:
            self.scaler.fit(balanced_X)

            # Apply PCA if required
            if self.use_pca:
                self.pca.fit(self.scaler.transform(balanced_X))

        return self

    def transform(self, X):
        # Drop unnecessary columns
        X_transformed = X.copy()
        for col in self.columns_to_drop:
            if col in X_transformed.columns:
                X_transformed.drop(columns=[col], inplace=True)

        # Apply scaling and PCA
        if self.use_scaler:
            X_transformed = self.scaler.transform(X_transformed)
            if self.use_pca:
                X_transformed = self.pca.transform(X_transformed)

        return X_transformed

    def fit_transform(self, user_labels, time_labels, X):
        self.fit(user_labels, time_labels, X)
        return self.transform(X)
