import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv", type=str, default="../result/STUV/Normal/nn_replay_aware_replay_prob_df.csv"
    )
    args = parser.parse_args()
    return args.csv


# Define function to compute EER and its threshold
def compute_eer_and_threshold(fpr, tpr, thresholds):
    """Compute the EER (Equal Error Rate) and its threshold given FPR, TPR, and thresholds."""
    fnr = 1 - tpr
    eer_index = np.nanargmin(np.abs(fpr - fnr))
    return fpr[eer_index], thresholds[eer_index]


if __name__ == "__main__":
    # Dictionary to store rejection rates for each user
    data_path = argparser()
    data_nn = pd.read_csv(data_path)
    rejection_rates_all_users = {timestamp: [] for timestamp in range(1, 8)}

    # Iterate over all users and calculate EER threshold and then rejection rates
    for user in data_nn["true_user"].unique():
        # Calculate EER threshold for the user for time=8 (True User vs. Test User)
        user_data = data_nn[data_nn["true_user"] == user]
        user_data_time_8 = user_data[user_data["time"] == 8]
        labels = (user_data_time_8["true_user"] == user_data_time_8["test_user"]).astype(int)
        probs = user_data_time_8["prob1"]
        fpr, tpr, thresholds = roc_curve(labels, probs)
        inter_fpr = np.linspace(0, 1, 100)
        interp_tpr = np.interp(inter_fpr, fpr, tpr)
        interp_thresholds = np.interp(inter_fpr, fpr, thresholds)
        # _, eer_threshold_user = compute_eer_and_threshold(fpr, tpr, thresholds)
        _, eer_threshold_user = compute_eer_and_threshold(inter_fpr, interp_tpr, interp_thresholds)

        # Compute replay attack rejection rate for each session using the EER threshold of the user
        for timestamp in range(1, 8):
            session_data = user_data[user_data["time"] == timestamp]
            # Reject if probability is below the threshold
            rejected = (session_data["prob1"] < eer_threshold_user).sum()
            total_attacks = len(session_data)
            rejection_rate = rejected / total_attacks
            rejection_rates_all_users[timestamp].append(rejection_rate)

    # Compute average rejection rates across all users
    average_rejection_rates = {
        timestamp: np.mean(rates) for timestamp, rates in rejection_rates_all_users.items()
    }

    # List to store EER values for each user
    eer_values_all_users = []

    # Iterate over all users and calculate EER
    for user in data_nn["true_user"].unique():
        # Calculate EER for the user for time=8 (True User vs. Test User)
        user_data = data_nn[data_nn["true_user"] == user]
        user_data_time_8 = user_data[user_data["time"] == 8]
        labels = (user_data_time_8["true_user"] == user_data_time_8["test_user"]).astype(int)
        probs = user_data_time_8["prob1"]
        fpr, tpr, thresholds = roc_curve(labels, probs)
        eer_value, _ = compute_eer_and_threshold(fpr, tpr, thresholds)
        eer_values_all_users.append(eer_value)

    # Compute average EER across all users
    average_eer_all_users = np.mean(eer_values_all_users)

    print("Average EER across all users: ", average_eer_all_users)
    print("Average rejection rates across all users: ", average_rejection_rates)
