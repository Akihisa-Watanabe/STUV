import os
import random

import numpy as np
import optuna
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, TimeSeriesSplit
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

np.random.seed(42)
random.seed(42)


class BasicMatcher:
    def __init__(self, model_type="logistic"):
        self.disable_parallel = os.getppid() != 1
        self.model_type = model_type
        if model_type == "logistic":
            self.base_model = LogisticRegression(max_iter=10000, random_state=42, multi_class="ovr")
            self.params = {
                "C": [0.001, 0.01, 0.1, 1, 10, 100],
                "penalty": ["l1", "l2", "elasticnet"],
            }
        elif model_type == "svm":
            self.base_model = SVC(probability=True, random_state=42)
            self.params = {
                "C": [0.001, 0.01, 0.1, 1, 10],
                "gamma": [0.001, 0.01, 0.1, 1, 10],
            }
        elif model_type == "neural":
            self.base_model = MLPClassifier(max_iter=10000, random_state=42)
            self.params = {
                "hidden_layer_sizes": [
                    (64, 128, 64),
                    (64, 64, 64),
                    (128, 128, 128),
                    (64, 128, 128),
                ],
                "activation": ["relu", "tanh"],
            }
        else:
            raise ValueError(
                "Currently, 'logistic', 'svm', and 'neural' model types are supported."
            )

        self.model = None

    def _balance_data(self, X_genuine, X_classifier, classifier_user_labels):
        """
        This function balances the data based on the provided approach:
        - Match the count of genuine_data
        - Within classifier data, ensure each user's data is balanced
        """
        N = X_genuine.shape[0]
        unique_users = np.unique(classifier_user_labels)
        U = len(unique_users)
        samples_per_user = N // U

        # Sampling data points from each unique user
        balanced_X_classifier = []
        for user in unique_users:
            user_indices = np.where(classifier_user_labels == user)[0]
            selected_indices = np.random.choice(user_indices, samples_per_user, replace=False)
            balanced_X_classifier.append(X_classifier[selected_indices])

        # Checking if we need to randomly select remaining samples to match the count
        remaining_samples = N - (samples_per_user * U)
        if remaining_samples > 0:
            excluded_indices = np.concatenate(balanced_X_classifier)
            available_indices = np.setdiff1d(np.arange(X_classifier.shape[0]), excluded_indices)
            np.random.seed(42)
            additional_indices = np.random.choice(
                available_indices, remaining_samples, replace=False
            )
            balanced_X_classifier.append(X_classifier[additional_indices])

        return np.vstack(balanced_X_classifier)

    def fit(self, X_genuine, X_classifier, classifier_user_labels):
        X_classifier_balanced = self._balance_data(X_genuine, X_classifier, classifier_user_labels)
        X_train = np.vstack((X_genuine, X_classifier_balanced))
        y_train = np.hstack((np.ones(X_genuine.shape[0]), np.zeros(X_classifier_balanced.shape[0])))

        # Setting up GridSearchCV
        np.random.seed(42)
        cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
        grid_search_params = {
            "estimator": self.base_model,
            "param_grid": self.params,
            "scoring": "f1",
            "cv": cv,
            "n_jobs": -1,
        }
        if self.disable_parallel:
            grid_search_params["n_jobs"] = 1

        grid_search = GridSearchCV(**grid_search_params)
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_

    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.

        Parameters:
        - X : array-like of shape (n_samples, n_features)
            Test samples.

        Returns:
        - probs : array-like of shape (n_samples, n_classes)
            The class probabilities of the samples for each class in the model.
        """
        return self.model.predict_proba(X)

    def score(self, X, y):
        """
        Returns the mean accuracy on the given test data and labels.

        Parameters:
        - X : array-like of shape (n_samples, n_features)
            Test samples.
        - y : array-like of shape (n_samples,)
            True labels for X.

        Returns:
        - score : float
            Mean accuracy of self.model on X and y.
        """
        return self.model.score(X, y)

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters:
        - X : array-like of shape (n_samples, n_features)
            Test samples.

        Returns:
        - y_pred : array-like of shape (n_samples,)
            Predicted class labels for X.
        """
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.

        Parameters:
        - X : array-like of shape (n_samples, n_features)
            Test samples.

        Returns:
        - probs : array-like of shape (n_samples, n_classes)
            The class probabilities of the samples for each class in the model.
        """
        return self.model.predict_proba(X)

    def score(self, X, y):
        return self.model.score(X, y)

    def evaluate(self, X, y):
        """
        Evaluates the model on the given test data and labels.

        Parameters:
        - X : array-like of shape (n_samples, n_features)
            Test samples.
        - y : array-like of shape (n_samples,)
            True labels for X.

        Returns:
        - accuracy : float
            Mean accuracy of self.model on X and y.
        - Recall : float
            Recall score of self.model on X and y.
        - F1 : float
            F1 score of self.model on X and y.

        """
        y_pred = self.predict(X)
        accuracy = self.score(X, y)
        recall = recall_score(y, y_pred)
        precision = precision_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        auc = roc_auc_score(y, y_pred)
        return accuracy, recall, precision, f1, auc

    def confusion_matrix(self, X, y):
        """
        Returns the confusion matrix of the model on the given test data and labels.

        Parameters:
        - X : array-like of shape (n_samples, n_features)
            Test samples.
        - y : array-like of shape (n_samples,)
            True labels for X.

        Returns:
        - confusion_matrix : array-like of shape (n_classes, n_classes)
            Confusion matrix of self.model on X and y.
        """
        return confusion_matrix(y, self.predict(X))


class STUVMatcher:
    """
    STUVMatcher is a class responsible for matching and liveness detection.

    Parameters:
    -----------
    sigma: float
        The standard deviation parameter.
    delta_t: float
        The delta time parameter.
    model_type: str
        The type of matcher model to use ("logistic", "svm", "neural").
    regression_type: str
        The type of regression model to use ("kernel_ridge").
    **kwargs: dict
        Additional keyword arguments.

    Attributes:
    -----------
    disable_parallel: bool
        Flag to disable parallel computation.
    sigma: float
        The standard deviation parameter.
    delta_t: float
        The delta time parameter.
    model_type: str
        The type of matcher model to use.
    regression_type: str
        The type of regression model to use.
    matcher_model: object
        The matcher model instance.
    params: dict
        The hyperparameters for the matcher model.
    """

    def __init__(
        self, sigma=1, delta_t=0.5, model_type="logistic", regression_type="kernel_ridge", **kwargs
    ):
        self.disable_parallel = os.getppid() != 1
        self.sigma = sigma
        self.Delta_t = delta_t
        self.model_type = model_type
        self.regression_type = regression_type
        self._set_matcher_model()

    def _set_matcher_model(self):
        """
        Initialize the matcher model based on the model_type attribute.
        """
        # Define model settings for each type of model.
        model_settings = {
            "logistic": {
                "model": LogisticRegression(max_iter=10000, random_state=42),
                "params": {
                    "C": [0.001, 0.01, 0.1, 1, 10, 100],
                    "penalty": ["l1", "l2", "elasticnet"],
                },
            },
            "svm": {
                "model": SVC(probability=True),
                "params": {"C": [0.001, 0.01, 0.1, 1, 10], "gamma": [0.001, 0.01, 0.1, 1, 10]},
            },
            "neural": {
                "model": MLPClassifier(max_iter=10000, random_state=42),
                "params": {"hidden_layer_sizes": [(64, 128, 64)], "activation": ["relu", "tanh"]},
            },
        }

        # Validate the model type and initialize the matcher model.
        if self.model_type not in model_settings:
            raise ValueError("Invalid model type. Choose from 'logistic', 'svm', or 'neural'.")

        self.matcher_model = model_settings[self.model_type]["model"]
        self.params = model_settings[self.model_type]["params"]

    def fit_matcher(self, X_genuine, X_classifier, classifier_user_labels):
        """
        Fit the matcher model based on genuine and classifier data.

        Parameters:
        -----------
        X_genuine: array-like
            The genuine data samples.
        X_classifier: array-like
            The classifier data samples.
        classifier_user_labels: array-like
            The labels corresponding to the classifier data samples.
        """
        # Balance the data and prepare the training set.
        X_classifier_balanced = self._balance_data(X_genuine, X_classifier, classifier_user_labels)

        # Combining the data
        X_train = np.vstack((X_genuine, X_classifier_balanced))
        y_train = np.hstack((np.ones(X_genuine.shape[0]), np.zeros(X_classifier_balanced.shape[0])))

        # Setting up GridSearchCV

        np.random.seed(42)
        cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
        grid_search_params = {
            "estimator": self.matcher_model,
            "param_grid": self.params,
            "scoring": "f1",
            "cv": cv,
            "n_jobs": -1,
        }
        if self.disable_parallel:
            grid_search_params["n_jobs"] = 1

        grid_search = GridSearchCV(**grid_search_params)
        grid_search.fit(X_train, y_train)
        self.matcher_model = grid_search.best_estimator_

    def fit_liveness(self, X_genuine, time_labels):
        """
        Fit the liveness model based on genuine data and time labels.

        Parameters:
        -----------
        X_genuine: array-like
            The genuine data samples.
        time_labels: array-like
            The time labels corresponding to the genuine data samples.
        """
        self.templates = X_genuine
        self.timestamps = time_labels
        self.T = max(np.unique(self.timestamps))
        if self.regression_type == "kernel_ridge":
            self.reg_model = self._train_kernel_ridge()
            self.sigma = self.optimize_parameters()
        else:
            raise ValueError(
                "Invalid regression type. Choose 'gaussian_process' or 'kernel_ridge'."
            )

    def _train_kernel_ridge(self):
        """
        Train the Kernel Ridge Regression model using GridSearchCV.

        Returns:
        --------
        object: Trained Kernel Ridge Regression model.
        """
        kr = KernelRidge(kernel="rbf")

        # Hyperparameters to be tuned using GridSearchCV
        param_grid = {"alpha": [0.1, 1e-2], "gamma": np.logspace(-3, 0, 10)}
        tscv = TimeSeriesSplit(n_splits=7)
        grid_search_params = {"estimator": kr, "param_grid": param_grid, "cv": tscv, "n_jobs": -1}
        if self.disable_parallel:
            grid_search_params["n_jobs"] = 1

        grid_search = GridSearchCV(**grid_search_params)
        grid_search.fit(self.timestamps, self.templates)
        return grid_search.best_estimator_

    def optimize_parameters(self):
        """
        Optimize the sigma and Delta_t parameters for the liveness model.

        Returns:
        --------
        tuple: optimized sigma and Delta_t values.
        """
        # Extract the latest templates from the data.
        unique_timestamps = np.unique(self.timestamps)
        self.latest_timestamp = unique_timestamps[-1]
        mask_latest = self.timestamps.flatten() == self.latest_timestamp
        self.latest_templates = self.templates[mask_latest]

        # Update sigma based on the standard deviation of the latest templates
        self.sigma = np.std(self.latest_templates)

        return self.sigma

    def _balance_data(self, X_genuine, X_classifier, classifier_user_labels):
        """
        This function balances the data based on the provided approach:
        - Match the count of genuine_data
        - Within classifier data, ensure each user's data is balanced
        """
        N = X_genuine.shape[0]
        unique_users = np.unique(classifier_user_labels)
        U = len(unique_users)
        samples_per_user = N // U

        # Sampling data points from each unique user
        balanced_X_classifier = []
        for user in unique_users:
            user_indices = np.where(classifier_user_labels == user)[0]
            selected_indices = np.random.choice(user_indices, samples_per_user, replace=False)
            balanced_X_classifier.append(X_classifier[selected_indices])

        # Checking if we need to randomly select remaining samples to match the count
        remaining_samples = N - (samples_per_user * U)
        if remaining_samples > 0:
            excluded_indices = np.concatenate(balanced_X_classifier)
            available_indices = np.setdiff1d(np.arange(X_classifier.shape[0]), excluded_indices)
            additional_indices = np.random.choice(
                available_indices, remaining_samples, replace=False
            )
            balanced_X_classifier.append(X_classifier[additional_indices])

        return np.vstack(balanced_X_classifier)

    def fit(self, X_genuine, time_labels, X_classifier, classifier_user_labels):
        """
        Fit both the matcher and liveness models.

        Parameters:
        -----------
        X_genuine: array-like
            The genuine data samples.
        time_labels: array-like
            The time labels corresponding to the genuine data samples.
        X_classifier: array-like
            The classifier data samples.
        classifier_user_labels: array-like
            The labels corresponding to the classifier data samples.
        """
        self.fit_matcher(X_genuine, X_classifier, classifier_user_labels)
        self.fit_liveness(X_genuine, time_labels)

    def predict(self, X, T):
        """
        Make a prediction based on the provided data and time T.

        Parameters:
        -----------
        X: array-like
            The data samples.
        T: float
            The current time.

        Returns:
        --------
        array-like: Prediction results (True or False).
        """
        proba_u = self._predict_proba_u(X)
        proba_t = self._predict_proba_t(X, T)

        proba = proba_u * proba_t
        return proba > 0.5

    def predict_proba(self, X, T):
        """
        Compute the prediction probabilities based on the provided data and time T.

        Parameters:
        -----------
        X: array-like
            The data samples.
        T: float
            The current time.

        Returns:
        --------
        array-like: Computed prediction probabilities.
        """
        proba_u = self._predict_proba_u(X)
        proba_t = self._predict_proba_t(X, T)
        proba_t = np.hstack([1 - proba_t, proba_t])
        proba = proba_u * proba_t

        return proba

    def _predict_proba_t(self, X, T):
        """
        Compute the probability for the liveness model.

        Parameters:
        -----------
        X_T: array-like
            The data samples at the current time T.
        x_t: array-like
            The predicted templates over time.

        Returns:
        --------
        array-like: computed probabilities.
        """
        self.query_time = np.linspace(0, T, 10 * len(self.timestamps)).reshape(
            -1, 1
        )  # np.linspace(0, T, 10).reshape(-1,1)#n
        query = X
        self.T = T
        self.predicted_templates = self._predict_trajectory(self.query_time)
        proba_t = self._calc_proba_t(query, self.predicted_templates)
        return proba_t

    def _predict_proba_u(self, X):
        """
        Compute the prediction probabilities for the matcher model.

        Parameters:
        -----------
        X: array-like
            The data samples.

        Returns:
        --------
        array-like: Computed prediction probabilities for the matcher model.
        """
        return self.matcher_model.predict_proba(X)

    def _predict_trajectory(self, query_time):
        """
        Predict the trajectory based on the query time using the trained regression model.

        Parameters:
        -----------
        query_time: array-like
            The query time points.

        Returns:
        --------
        array-like: Predicted templates at the query time points.
        """
        return self.reg_model.predict(query_time)

    def _calc_proba_t(self, X_T, x_t):
        """
        Compute the probability for the liveness model.

        Parameters:
        -----------
        X_T: array-like
            The data samples at the current time T.
        x_t: array-like
            The predicted templates over time.

        Returns:
        --------
        array-like: computed probabilities.
        """
        delta = self.query_time[1] - self.query_time[0]
        delta_n = int(self.Delta_t / delta)
        delta_time = self.query_time[-delta_n:-1]
        x_t_last = x_t[-delta_n:-1, :]

        tmp_numerator = np.exp(
            -np.linalg.norm(X_T[:, np.newaxis] - x_t_last, axis=2) ** 2 / (2 * self.sigma**2)
        )
        tmp_denominator = np.exp(
            -np.linalg.norm(X_T[:, np.newaxis] - x_t, axis=2) ** 2 / (2 * self.sigma**2)
        )
        numerator = np.trapz(tmp_numerator, x=delta_time.T, axis=1).reshape(-1, 1)
        denominator = np.trapz(tmp_denominator, x=self.query_time.T, axis=1).reshape(-1, 1)
        # Handle the division by zero and NaN cases
        with np.errstate(divide="ignore", invalid="ignore"):
            P = np.nan_to_num(numerator / denominator, nan=1e-7)
        return P
