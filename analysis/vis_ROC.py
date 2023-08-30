import argparse
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

# fix seed
np.random.seed(42)
random.seed(42)


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv", type=str, default="../result/STUV/Normal/nn_replay_aware_replay_prob_df.csv"
    )
    args = parser.parse_args()
    return args.csv


# Function to calculate tprs for each timestamp pair based on the provided code
def calc_tpr_for_timestamp(data, timestamp):
    all_users = np.unique(data["true_user"])
    mean_fpr = np.linspace(0, 1, 30)
    tprs_list = []

    for user in all_users:
        user_data = data[data["true_user"] == user]
        user_data = user_data[user_data["true_user"] == user_data["test_user"]]

        labels = (
            user_data["time"]
            .apply(lambda x: 1 if x == 8 else (0 if x == timestamp else None))
            .dropna()
        )
        probs = user_data.loc[labels.index, "prob1"]
        fpr, tpr, _ = roc_curve(labels, probs)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs_list.append(interp_tpr)

    return np.mean(tprs_list, axis=0)


def calc_tpr_for_true_user(data):
    all_users = np.unique(data["true_user"])
    mean_fpr = np.linspace(0, 1, 100)
    tprs_list = []

    for user in all_users:
        user_data = data[data["true_user"] == user]
        user_data = user_data[user_data["time"] == 8]

        labels = (user_data["true_user"] == user_data["test_user"]).astype(int)
        probs = user_data["prob1"]
        fpr, tpr, _ = roc_curve(labels, probs)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs_list.append(interp_tpr)

    return np.mean(tprs_list, axis=0)


if __name__ == "__main__":
    data_path = argparser()
    data = pd.read_csv(data_path)
    # Plotting the ROC curves for each timestamp pair
    plt.rcParams["axes.axisbelow"] = True
    plt.rcParams["font.family"] = "Times New Roman"
    fig, ax = plt.subplots(tight_layout=True)
    ax.grid(True, which="both", axis="both", linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)

    mean_fpr = np.linspace(0, 1, 30)
    colors = plt.cm.viridis(np.linspace(0, 1, 7))

    for timestamp in range(1, 7):
        tpr = calc_tpr_for_timestamp(data, timestamp)
        ax.plot(mean_fpr, tpr, color=colors[timestamp - 1], label=f"t={timestamp} vs t=7", lw=2)

    # Calculate tpr for true user vs all test users
    tpr_true_user = calc_tpr_for_true_user(data)
    mean_fpr = np.linspace(0, 1, 100)
    ax.plot(mean_fpr, tpr_true_user, color="red", label="t=8 vs All Test Users", lw=2)

    ax.set_xlabel("False Positive Rate", fontsize=16)
    ax.set_ylabel("True Positive Rate", fontsize=16)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.savefig("./ROC.png", dpi=300)
    plt.show()
