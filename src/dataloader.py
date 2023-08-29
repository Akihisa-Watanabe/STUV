import pandas as pd


class KeystrokeDataLoader:
    """
    Load the DSL-StrongPasswordData dataset and provide utility functions to get data.
    """

    def __init__(
        self, data, genuine_user, feature_users, classifier_users, imposter_users
    ):
        self.data = data
        self.genuine_user = genuine_user
        self.feature_users = feature_users
        self.classifier_users = classifier_users
        self.imposter_users = imposter_users

    def get_data_of_user(self, user):
        """Retrieve data of a specific user."""
        user_data = self.data[self.data["subject"] == user]
        return (
            user_data[["subject"]],
            user_data[["sessionIndex"]],
            user_data.drop(columns=["subject", "sessionIndex", "rep"]),
        )

    def get_data_of_users(self, users):
        """Retrieve data of multiple users."""
        users_data = self.data[self.data["subject"].isin(users)]
        return (
            users_data[["subject"]],
            users_data[["sessionIndex"]],
            users_data.drop(columns=["subject", "sessionIndex", "rep"]),
        )

    def get_data_of_user_at_time(self, user, t):
        """Retrieve data of a specific user at a specific session/time."""
        user_data = self.data[
            (self.data["subject"] == user) & (self.data["sessionIndex"] == t)
        ]
        return (
            user_data[["subject"]],
            user_data[["sessionIndex"]],
            user_data.drop(columns=["subject", "sessionIndex", "rep"]),
        )

    def get_data_of_users_at_time(self, users, t):
        """Retrieve data of multiple users at a specific session/time."""
        users_data = self.data[
            (self.data["subject"].isin(users)) & (self.data["sessionIndex"] == t)
        ]
        return (
            users_data[["subject"]],
            users_data[["sessionIndex"]],
            users_data.drop(columns=["subject", "sessionIndex", "rep"]),
        )

    def get_max_time_of_user(self, user):
        """Get the maximum session/time index for a specific user."""
        return self.data[self.data["subject"] == user]["sessionIndex"].max()

    def get_min_time_of_user(self, user):
        """Get the minimum session/time index for a specific user."""
        return self.data[self.data["subject"] == user]["sessionIndex"].min()

    def get_all_data(self):
        """Retrieve the entire dataset."""
        return (
            self.data[["subject"]],
            self.data[["sessionIndex"]],
            self.data.drop(columns=["subject", "sessionIndex", "rep"]),
        )
