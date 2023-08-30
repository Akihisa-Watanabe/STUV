import argparse
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

# fix seed
np.random.seed(42)
random.seed(42)


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Normal", help="time")
    args = parser.parse_args()
    return args.model


# Calculate scores for each user using scikit-learn and then compute the average
def compute_user_scores(df):
    # True labels: 1 if true_user is equal to test_user, 0 otherwise
    true_labels = (df["true_user"] == df["test_user"]).astype(int).values
    scores = df["prob1"].values

    precision, recall, _ = precision_recall_curve(true_labels, scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)

    fpr, tpr, _ = roc_curve(true_labels, scores)
    eer = fpr[np.nanargmin(np.absolute(fpr - (1 - tpr)))]

    # Return the average recall, average F1, and EER for the user
    return np.mean(recall), np.mean(f1_scores), eer


def compute_prob(df):
    # True labels: 1 if true_user is equal to test_user, 0 otherwise
    true_data = df[df["true_user"] == df["test_user"]]
    scores = true_data["prob1"].values
    # 四捨五入して2けたにする
    return np.mean(scores)


def visualize(model, senario, ax, i):
    # df = #pd.read_csv('./result/{}/{}/nn_mather_df.csv'.format(senario,model))
    if senario == "Static":
        df = pd.read_csv("../result/Static/Normal/mather_df.csv")
    elif senario == "Adaptive":
        df = pd.read_csv("../result/Adaptive/Normal/mather_df.csv")
    average_recalls = []
    average_f1s = []
    eers = []
    probs = []

    # Iterate over each unique timestamp
    for time in df["time"].unique():
        time_df = df[df["time"] == time]
        recalls = []
        f1s = []
        user_eers = []
        probs_u = []

        # Iterate over each unique user for the current timestamp
        for user in time_df["true_user"].unique():
            user_df = time_df[time_df["true_user"] == user]
            recall, f1, eer = compute_user_scores(user_df)
            m_prob = compute_prob(user_df)

            recalls.append(recall)
            f1s.append(f1)
            user_eers.append(eer)
            probs_u.append(m_prob)

        # Compute the average for the current timestamp
        average_recalls.append(np.mean(recalls))
        average_f1s.append(np.mean(f1s))
        eers.append(np.mean(user_eers))
        probs.append(np.round(np.mean(probs_u), 2))

    # ax.plot(df['time'].unique(), average_recalls, label='{} Verification'.format(senario), marker='o')
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    ax.scatter(
        df["time"].unique() - 1,
        probs,
        label="{} Verification Model".format(senario),
        facecolors="white",
        edgecolors=colors[i],
        s=80,
        zorder=2,
    )
    ax.scatter(df["time"].unique() - 1, probs, s=2, color=colors[i], zorder=2)
    ax.plot(df["time"].unique() - 1, probs, zorder=1)
    # ax.plot(df['time'].unique(), average_f1s, label='Average F1', marker='x')
    # ax.plot(df['time'].unique(), eers, label='EER', marker='s', color='green')
    # print(probs)


if __name__ == "__main__":
    np.random.seed(42)
    model = argparser()
    senarios = ["Static", "Adaptive"]

    plt.rcParams["axes.axisbelow"] = True
    plt.rcParams["font.family"] = "Times New Roman"
    fig, ax = plt.subplots(tight_layout=True, figsize=(6, 3))
    ax.grid(True, which="both", axis="both", linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)
    for i, senario in enumerate(senarios):
        visualize(model, senario, ax, i)

    plt.xlabel("Time", fontsize=14)
    plt.ylim([0.7, 1])
    plt.ylabel("Probability", fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig("./analysis_time_score.png", dpi=300)
    # plt.show()
