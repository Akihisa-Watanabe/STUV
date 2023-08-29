import random

import numpy as np
import optuna
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, TimeSeriesSplit
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.utils import resample, shuffle

np.random.seed(42)
random.seed(42)


class BasicMatcher:
    def __init__(self, model_type="logistic"):
        self.model_type = model_type
        if model_type == "logistic":
            self.base_model = LogisticRegression(
                max_iter=10000, random_state=42, multi_class="ovr"
            )
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
            selected_indices = np.random.choice(
                user_indices, samples_per_user, replace=False
            )
            balanced_X_classifier.append(X_classifier[selected_indices])

        # Checking if we need to randomly select remaining samples to match the count
        remaining_samples = N - (samples_per_user * U)
        if remaining_samples > 0:
            excluded_indices = np.concatenate(balanced_X_classifier)
            available_indices = np.setdiff1d(
                np.arange(X_classifier.shape[0]), excluded_indices
            )
            np.random.seed(42)
            additional_indices = np.random.choice(
                available_indices, remaining_samples, replace=False
            )
            balanced_X_classifier.append(X_classifier[additional_indices])

        return np.vstack(balanced_X_classifier)

    def fit(self, X_genuine, X_classifier, classifier_user_labels):
        # Setting up GridSearchCV
        np.random.seed(42)
        # print("Fitting the model...")
        # X_genuine
        # print("X_genuine shape: {}".format(X_genuine.shape))
        cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
        grid_search = GridSearchCV(
            self.base_model, self.params, scoring="f1", cv=cv, n_jobs=-1
        )

        X_classifier_balanced = self._balance_data(
            X_genuine, X_classifier, classifier_user_labels
        )
        X_train = np.vstack((X_genuine, X_classifier_balanced))
        y_train = np.hstack(
            (np.ones(X_genuine.shape[0]), np.zeros(X_classifier_balanced.shape[0]))
        )
        grid_search.fit(X_train, y_train)
        # print best f1 score
        # print("Best F1 score: {:.4f}".format(grid_search.best_score_))
        # print("Best parameters: {}".format(grid_search.best_params_))
        # Using the best estimator
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
    def __init__(
        self, sigma=1, model_type="logistic", regression_type="kernel_ridge", **kwargs
    ):
        self.sigma = sigma
        self.model_type = model_type
        self.regression_type = regression_type
        if model_type == "logistic":
            self.mather_model = LogisticRegression(max_iter=10000, random_state=42)
            self.params = {
                "C": [0.001, 0.01, 0.1, 1, 10, 100],
                "penalty": ["l1", "l2", "elasticnet"],
            }
        elif model_type == "svm":
            self.mather_model = SVC(probability=True)
            self.params = {
                "C": [0.001, 0.01, 0.1, 1, 10],
                "gamma": [0.001, 0.01, 0.1, 1, 10],
            }
        elif model_type == "neural":
            self.mather_model = MLPClassifier(max_iter=10000, random_state=42)
            self.params = {
                "hidden_layer_sizes": [(64, 128, 64)],
                "activation": ["relu", "tanh"],
            }
        else:
            raise ValueError(
                "Currently, 'logistic', 'svm', and 'neural' model types are supported."
            )

    def fit_matcher(self, X_genuine, X_classifier, classifier_user_labels):
        # Balancing the data
        X_classifier_balanced = self._balance_data(
            X_genuine, X_classifier, classifier_user_labels
        )

        # Combining the data
        X_train = np.vstack((X_genuine, X_classifier_balanced))
        y_train = np.hstack(
            (np.ones(X_genuine.shape[0]), np.zeros(X_classifier_balanced.shape[0]))
        )

        # Setting up GridSearchCV
        np.random.seed(42)
        cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
        grid_search = GridSearchCV(
            self.mather_model, self.params, scoring="f1", cv=3, n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        # Using the best estimator
        self.mather_model = grid_search.best_estimator_

    def fit_liveness(self, X_genuine, time_labels):
        self.templates = X_genuine
        self.timestamps = time_labels
        self.T = max(np.unique(self.timestamps))
        if self.regression_type == "kernel_ridge":
            self.reg_model = self._train_kernel_ridge()
            self.sigma = self.optimize_parameters(n_trials=1000)
        else:
            raise ValueError(
                "Invalid regression type. Choose 'gaussian_process' or 'kernel_ridge'."
            )

    def _train_kernel_ridge(self):
        """Train the Kernel Ridge Regression model with GridSearchCV."""
        kr = KernelRidge(kernel="rbf")

        # Hyperparameters to be tuned using GridSearchCV
        param_grid = {"alpha": [0.1, 1e-2], "gamma": np.logspace(-3, 0, 10)}
        tscv = TimeSeriesSplit(n_splits=7)
        grid_search = GridSearchCV(kr, param_grid, cv=tscv, n_jobs=-1)
        grid_search.fit(self.timestamps, self.templates)
        return grid_search.best_estimator_

    def optimize_parameters(self, n_trials=1000):
        """Optimize sigma_d and sigma_w using Optuna."""
        # Store the original templates before optimization
        original_templates = self.templates.copy()
        original_timestamps = self.timestamps.copy()
        original_T = self.T.copy()
        # Separate out the latest timestamp templates and the rest
        unique_timestamps = np.unique(self.timestamps)
        unique_timestamps.sort()
        self.latest_timestamp = unique_timestamps[-1]
        mask_latest = (
            self.timestamps.flatten() == self.latest_timestamp
        )  # Flatten self.timestamps

        self.latest_templates = self.templates[mask_latest]
        self.templates = self.templates[~mask_latest]
        self.timestamps = self.timestamps[~mask_latest]
        self.unique_timestamps = np.unique(self.timestamps)
        study = optuna.create_study(
            direction="minimize", sampler=optuna.samplers.TPESampler(seed=7)
        )
        study.optimize(self._objective, n_trials=n_trials, n_jobs=1)

        # Revert the self.templates back to its original state
        self.templates = original_templates
        self.timestamps = original_timestamps
        self.T = original_T
        self.sigma = study.best_params["sigma"]

        return study.best_params["sigma"]

    def _objective(self, trial):
        """Objective function for Optuna optimization."""
        self.sigma = trial.suggest_float("sigma", 0.1, 10)
        latest_p = self._predict_proba_t(self.latest_templates, self.latest_timestamp)
        objective_value = np.sum(-np.log(latest_p + 1e-5))
        for timestamp in self.unique_timestamps:
            mask_timestamp = self.timestamps.flatten() == timestamp
            t_templates = self.templates[mask_timestamp]
            t_p = self._predict_proba_t(t_templates, self.latest_timestamp)
            cross_entropy = -np.sum(np.log(1 - t_p + 1e-5))
            objective_value += cross_entropy
        return objective_value

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
            selected_indices = np.random.choice(
                user_indices, samples_per_user, replace=False
            )
            balanced_X_classifier.append(X_classifier[selected_indices])

        # Checking if we need to randomly select remaining samples to match the count
        remaining_samples = N - (samples_per_user * U)
        if remaining_samples > 0:
            excluded_indices = np.concatenate(balanced_X_classifier)
            available_indices = np.setdiff1d(
                np.arange(X_classifier.shape[0]), excluded_indices
            )
            additional_indices = np.random.choice(
                available_indices, remaining_samples, replace=False
            )
            balanced_X_classifier.append(X_classifier[additional_indices])

        return np.vstack(balanced_X_classifier)

    def fit(self, X_genuine, time_labels, X_classifier, classifier_user_labels):
        self.fit_matcher(X_genuine, X_classifier, classifier_user_labels)
        self.fit_liveness(X_genuine, time_labels)

    def predict(self, X, T):
        proba_u = self._predict_proba_u(X)
        proba_t = self._predict_proba_t(X, T)

        proba = proba_u * proba_t
        return proba > 0.5

    def predict_proba(self, X, T):
        proba_u = self._predict_proba_u(X)
        proba_t = self._predict_proba_t(X, T)
        proba_t = np.hstack([1 - proba_t, proba_t])
        proba = proba_u * proba_t

        return proba

    def _predict_proba_t(self, X, T):
        self.query_time = np.linspace(0, T, 10 * len(self.timestamps)).reshape(
            -1, 1
        )  # np.linspace(0, T, 10).reshape(-1,1)#n
        query = X
        self.T = T
        self.predicted_templates = self._predict_trajectory(self.query_time)
        proba_t = self._calc_proba_t(query, self.predicted_templates)
        return proba_t

    def _predict_proba_u(self, X):
        return self.mather_model.predict_proba(X)

    def _predict_trajectory(self, query_time):
        return self.reg_model.predict(query_time)

    def _calc_proba_t(self, X_T, x_t):
        """Compute the Probability."""
        delta_t = self.query_time[1] - self.query_time[0]
        delta_n = int(0.5 / delta_t)
        delta_time = self.query_time[-delta_n:-1]
        x_t_last = x_t[-delta_n:-1, :]

        tmp_numerator = np.exp(
            -np.linalg.norm(X_T[:, np.newaxis] - x_t_last, axis=2) ** 2
            / (2 * self.sigma**2)
        )
        tmp_denominator = np.exp(
            -np.linalg.norm(X_T[:, np.newaxis] - x_t, axis=2) ** 2
            / (2 * self.sigma**2)
        )
        numerator = np.trapz(tmp_numerator, x=delta_time.T, axis=1).reshape(-1, 1)
        denominator = np.trapz(tmp_denominator, x=self.query_time.T, axis=1).reshape(
            -1, 1
        )
        P = np.nan_to_num(numerator / denominator, nan=10e-7)
        return P
