# visualize latent space by using t-SNE and Gaussian Distribution
import argparse
import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Ellipse
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture


def draw_ellipse(position, covariance, ax=None, **kwargs):
    ax = ax or plt.gca()
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    for nsig in range(1, 2):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height, angle=angle, **kwargs))


def reduce_dim(features):
    # Compressing the features into 2 dimensions using t-SNE
    tsne = TSNE(n_components=2, random_state=0)
    features_2d = tsne.fit_transform(features)

    # # Creating a DataFrame for the compressed features
    features_2d_df = pd.DataFrame(data=features_2d, columns=["Dimension 1", "Dimension 2"])

    return features_2d_df


def visualize(user, final_df, save_dir):
    plt.rcParams["axes.axisbelow"] = True
    plt.rcParams["font.family"] = "Times New Roman"
    # Creating a figure to plot
    fig, ax = plt.subplots(tight_layout=True)

    # Getting unique time indices
    time_indices = final_df["time"].unique()

    # Color map for different time points
    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0, 1, len(time_indices)))

    # Initializing an empty list to store the means
    means = []

    # Plotting the features for each time point
    for index, color in zip(time_indices, colors):
        # Extracting the features for the current time index
        time_data = final_df[final_df["time"] == index][["Dimension 1", "Dimension 2"]]
        gmm = GaussianMixture(n_components=1).fit(time_data)
        mean = gmm.means_[0]
        covariance = gmm.covariances_[0]
        draw_ellipse(mean, covariance, ax=ax, alpha=0.2, color=color, zorder=1)
        ax.scatter(mean[0], mean[1], color=color, s=100, label=f"t={index-1}", zorder=3, marker="x")
        means.append(mean)

    # Drawing arrows between successive means to indicate the order of the sessions
    means = np.array(means)
    for i in range(len(means) - 1):
        if i == len(means) - 2:
            arrow = patches.ConnectionPatch(
                means[i],
                means[i + 1],
                "data",
                "data",
                arrowstyle="-|>",
                mutation_scale=20,
                lw=1.5,
                color="black",
                linestyle="--",
                zorder=3,
            )
            ax.add_artist(arrow)
        else:
            ax.plot(
                [means[i, 0], means[i + 1, 0]],
                [means[i, 1], means[i + 1, 1]],
                color="black",
                linestyle="dashed",
                zorder=3,
            )
    # Plotting the sample data points
    ax.scatter(
        final_df["Dimension 1"],
        final_df["Dimension 2"],
        c=final_df["time"],
        cmap="viridis",
        alpha=0.2,
        zorder=1,
        marker="x",
    )

    # Adding title and labels to the plot
    ax.set_xlabel("Dimension 1", fontsize=16)
    ax.set_ylabel("Dimension 2", fontsize=16)
    # Arrange the labels in a horizontal line.
    ax.legend(loc="upper right", fontsize=14)
    path = save_dir + "/latent_reprentation{}.png".format(user)
    plt.savefig(path, dpi=300)
    plt.close()


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Normal", help="time")
    parser.add_argument("--user", type=str, default="s003", help="time")
    args = parser.parse_args()
    return args.model, args.user


if __name__ == "__main__":
    model, user = argparser()

    features_path = "../result/Adaptive/{}/feature_df.csv".format(model)
    features_df = pd.read_csv(features_path)
    user_data = features_df[features_df["true_user"] == user]
    user_data = user_data[user_data["test_user"] == user]

    # Extracting features and labels
    features = user_data.drop(["true_user", "test_user", "time"], axis=1)
    labels = user_data["time"]
    # Compressing the features into 2 dimensions using t-SNE
    features_2d_df = reduce_dim(features)
    # Joining the labels with the compressed features
    final_df = pd.concat([features_2d_df, labels.reset_index(drop=True)], axis=1)
    # Visualizing the compressed features
    # save_dir = './result/{}/{}'.format(model,user)
    save_dir = "./"
    # # Create a directory to save the figures
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)

    visualize(user, final_df, save_dir)
