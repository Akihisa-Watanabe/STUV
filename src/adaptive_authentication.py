import gc
import random

import numpy as np
import pandas as pd
from tqdm import tqdm

from dataloader import KeystrokeDataLoader
from feature_extractor import BasicFeatureExtractor
from matcher import BasicMatcher
from util import prepare_user_configurations_updated

np.random.seed(42)
random.seed(42)


def authentication_scenario(data, genuine_user, m, t_now):
    # Initialize user configurations
    (
        genuine_user,
        feature_users,
        classifier_users,
        imposter_users,
    ) = prepare_user_configurations_updated(data, genuine_user, 0.2, 0.6, 0.2)

    # Initialize dataloader
    dataloader = KeystrokeDataLoader(
        data, genuine_user, feature_users, classifier_users, imposter_users
    )

    # Step 2: Learn feature extraction model F_sys using feature_users at t=1
    f_user_labels, f_time_labels, feature_data = dataloader.get_data_of_users_at_time(
        feature_users, t=1
    )
    feature_extractor = BasicFeatureExtractor()
    feature_extractor.fit(
        f_user_labels.squeeze(), f_time_labels.squeeze(), feature_data
    )
    # print('Feature Extraction Model F_sys learned successfully')

    # Step 3: Take m random samples from genuine user at t=0
    (
        genuine_user_labels,
        genuine_time_labels,
        genuine_data_t0,
    ) = dataloader.get_data_of_users_at_time([genuine_user], t=1)
    (
        imposter_user_labels,
        imposter_time_labels,
        imposter_data_t0,
    ) = dataloader.get_data_of_users_at_time(imposter_users, t=1)
    (
        classifier_user_labels,
        classifier_time_labels,
        classifier_data_t0,
    ) = dataloader.get_data_of_users_at_time(classifier_users, t=1)

    # Transforming the imposter data using the feature extractor
    T = pd.DataFrame()
    R = pd.DataFrame()

    T_indices = np.random.choice(genuine_data_t0.shape[0], m, replace=False)
    T_data = genuine_data_t0.iloc[T_indices]
    T_feature = feature_extractor.transform(T_data)
    T = pd.DataFrame(
        T_feature, columns=[f"feature{i}" for i in range(T_feature.shape[1])]
    )
    T["time"] = 1

    R_data = genuine_data_t0.drop(genuine_data_t0.index[T_indices])
    R_feature = feature_extractor.transform(R_data)
    R = pd.DataFrame(
        R_feature, columns=[f"feature{i}" for i in range(R_feature.shape[1])]
    )
    R["time"] = 1

    X_train_genuine = feature_extractor.transform(T_data)
    X_train_classifier = feature_extractor.transform(classifier_data_t0)

    X_test_genuine = feature_extractor.transform(R_data)
    X_test_imposter = feature_extractor.transform(imposter_data_t0)

    X_test = np.vstack((X_test_genuine, X_test_imposter))
    y_test = np.hstack(
        (np.ones(X_test_genuine.shape[0]), np.zeros(X_test_imposter.shape[0]))
    )
    X_test_imposter = feature_extractor.transform(imposter_data_t0)

    # Combining test data and labels
    X_test = np.vstack((X_test_genuine, X_test_imposter))
    y_test = np.hstack(
        (np.ones(X_test_genuine.shape[0]), np.zeros(X_test_imposter.shape[0]))
    )

    classifier_modified = BasicMatcher(model_type="neural")
    classifier_modified.fit(
        X_train_genuine, X_train_classifier, classifier_user_labels.squeeze()
    )
    # accuracy, recall, precision, f1, auc = classifier_modified.evaluate(X_test, y_test)
    # print(f"AUC of modified classifier at t=1: {auc}")

    # save features(test data) of each time
    feature_dataframes = []

    user_labels_1 = np.hstack(
        (
            genuine_user_labels.drop(genuine_user_labels.index[T_indices]).squeeze(),
            imposter_user_labels.squeeze(),
        )
    )  # Correcting the labels
    feature_df_t1 = pd.DataFrame(
        X_test, columns=[f"feature{i}" for i in range(X_test.shape[1])]
    )
    feature_df_t1["true_user"] = genuine_user
    feature_df_t1["test_user"] = user_labels_1
    feature_df_t1["time"] = 1
    feature_dataframes.append(feature_df_t1)

    # save predicted probabilities of each time
    prob_dataframes = []
    prob_df_t1 = pd.DataFrame(
        classifier_modified.predict_proba(X_test)[:, 1], columns=["prob1"]
    )
    prob_df_t1["true_user"] = genuine_user
    prob_df_t1["test_user"] = user_labels_1
    prob_df_t1["threshold"] = 0.5
    prob_df_t1["time"] = 1
    prob_dataframes.append(prob_df_t1)
    for t in range(2, t_now + 1):
        (
            genuine_user_labels_t,
            genuine_time_labels_t,
            genuine_data_t,
        ) = dataloader.get_data_of_users_at_time([genuine_user], t=t)
        (
            imposter_user_labels_t,
            imposter_time_labels_t,
            imposter_data_t,
        ) = dataloader.get_data_of_users_at_time(imposter_users, t=t)

        X_test_genuine_t = feature_extractor.transform(genuine_data_t)
        X_test_imposter_t = feature_extractor.transform(imposter_data_t)
        X_test_t = np.vstack((X_test_genuine_t, X_test_imposter_t))
        y_test_t = np.hstack(
            (np.ones(X_test_genuine_t.shape[0]), np.zeros(X_test_imposter_t.shape[0]))
        )
        accuracy, recall, precision, f1, auc = classifier_modified.evaluate(
            X_test_t, y_test_t
        )
        # print(f"AUC of modified classifier at t={t}: {auc}")
        feature_df_t = pd.DataFrame(
            X_test_t, columns=[f"feature{i}" for i in range(X_test_t.shape[1])]
        )
        feature_df_t["true_user"] = genuine_user
        feature_df_t["test_user"] = np.hstack(
            (genuine_user_labels_t.squeeze(), imposter_user_labels_t.squeeze())
        )
        feature_df_t["time"] = t
        feature_dataframes.append(feature_df_t)

        prob_df_t = pd.DataFrame(
            classifier_modified.predict_proba(X_test_t)[:, 1], columns=["prob1"]
        )
        prob_df_t["true_user"] = genuine_user
        prob_df_t["test_user"] = np.hstack(
            (genuine_user_labels_t.squeeze(), imposter_user_labels_t.squeeze())
        )
        prob_df_t["threshold"] = 0.5
        prob_df_t["time"] = t
        prob_dataframes.append(prob_df_t)

        # Model Updata
        if t < t_now:
            pred_proba_t = classifier_modified.predict_proba(X_test_genuine_t)
            threshold = 0.5
            high_proba_indices = np.where(pred_proba_t[:, 1] > threshold)[0]
            if len(high_proba_indices) <= m:
                selected_random_indices = high_proba_indices
            else:
                selected_random_indices = np.random.choice(
                    high_proba_indices, m, replace=False
                )

            selected_samples = genuine_data_t.iloc[selected_random_indices]
            selected_features = feature_extractor.transform(selected_samples)
            T_t = pd.DataFrame(
                selected_features,
                columns=[f"feature{i}" for i in range(selected_features.shape[1])],
            )
            T_t["time"] = t
            remaining_samples = genuine_data_t.drop(
                genuine_data_t.index[selected_random_indices]
            )
            remaining_features = feature_extractor.transform(remaining_samples)
            R_t = pd.DataFrame(
                remaining_features,
                columns=[f"feature{i}" for i in range(remaining_features.shape[1])],
            )
            R_t["time"] = t
            # Add selected samples to T
            T = pd.concat([T, T_t])
            R = pd.concat([R, R_t])
            # Retraining the classifier
            X_train_genuine_t = T.drop(["time"], axis=1).values
            classifier_modified.fit(
                X_train_genuine_t, X_train_classifier, classifier_user_labels.squeeze()
            )

    # Replay Attack
    replay_prob_dataframes = []
    replay_feature_dataframes = []
    for t in range(1, t_now + 1):
        if t < t_now:
            # get feature data of each time
            R_data_t = R[R["time"] == t]
            X_Replay_t = R_data_t.drop(["time"], axis=1).values

            replay_feature_df_t = pd.DataFrame(
                X_Replay_t, columns=[f"feature{i}" for i in range(X_Replay_t.shape[1])]
            )
            replay_feature_df_t["true_user"] = genuine_user
            replay_feature_df_t["test_user"] = genuine_user
            replay_feature_df_t["time"] = t
            replay_feature_dataframes.append(replay_feature_df_t)

            replay_prob_df_t = pd.DataFrame(
                classifier_modified.predict_proba(X_Replay_t)[:, 1], columns=["prob1"]
            )
            replay_prob_df_t["true_user"] = genuine_user
            replay_prob_df_t["test_user"] = genuine_user
            replay_prob_df_t["threshold"] = 0.5
            replay_prob_df_t["time"] = t
            replay_prob_dataframes.append(replay_prob_df_t)
        else:
            (
                genuine_user_labels_tnow,
                genuine_time_labels_tnow,
                genuine_data_tnow,
            ) = dataloader.get_data_of_users_at_time([genuine_user], t=t)

            X_test_genuine_tnow = feature_extractor.transform(genuine_data_tnow)
            replay_feature_df_t = pd.DataFrame(
                X_test_genuine_tnow,
                columns=[f"feature{i}" for i in range(X_test_genuine_tnow.shape[1])],
            )
            replay_feature_df_t["true_user"] = genuine_user
            replay_feature_df_t["test_user"] = genuine_user
            replay_feature_df_t["time"] = t
            replay_feature_dataframes.append(replay_feature_df_t)

            replay_prob_df_t = pd.DataFrame(
                classifier_modified.predict_proba(X_test_genuine_tnow)[:, 1],
                columns=["prob1"],
            )
            replay_prob_df_t["true_user"] = genuine_user
            replay_prob_df_t["test_user"] = genuine_user
            replay_prob_df_t["threshold"] = 0.5
            replay_prob_df_t["time"] = t
            replay_prob_dataframes.append(replay_prob_df_t)

            (
                imposter_user_labels_tnow,
                imposter_time_labels_tnow,
                imposter_data_tnow,
            ) = dataloader.get_data_of_users_at_time(imposter_users, t=t)
            X_test_imposter_tnow = feature_extractor.transform(imposter_data_tnow)
            replay_prob_df_t_imposter = pd.DataFrame(
                classifier_modified.predict_proba(X_test_imposter_tnow)[:, 1],
                columns=["prob1"],
            )
            replay_prob_df_t_imposter["true_user"] = genuine_user
            replay_prob_df_t_imposter["test_user"] = np.array(
                imposter_user_labels_tnow.squeeze()
            )
            replay_prob_df_t_imposter["threshold"] = 0.5
            replay_prob_df_t_imposter["time"] = t
            replay_prob_dataframes.append(replay_prob_df_t_imposter)

    # Concatenate all feature dataframes
    feature_df = pd.concat(feature_dataframes, ignore_index=True).reindex(
        columns=["true_user", "test_user", "time"]
        + [f"feature{i}" for i in range(X_test.shape[1])]
    )
    prob_df = pd.concat(prob_dataframes, ignore_index=True).reindex(
        columns=["true_user", "test_user", "time", "prob1"]
    )
    replay_feature_df = pd.concat(replay_feature_dataframes, ignore_index=True).reindex(
        columns=["true_user", "test_user", "time"]
        + [f"feature{i}" for i in range(X_test.shape[1])]
    )
    replay_prob_df = pd.concat(replay_prob_dataframes, ignore_index=True).reindex(
        columns=["true_user", "test_user", "time", "prob1"]
    )

    # remove all class instances
    del dataloader
    del feature_extractor
    del classifier_modified

    gc.collect()
    return feature_df, prob_df, replay_feature_df, replay_prob_df


if __name__ == "__main__":
    data = pd.read_csv("../data/DSL-StrongPasswordData.csv")
    m = 20
    t_now = 8
    all_users = data["subject"].unique()
    feature_df = pd.DataFrame()
    mather_df = pd.DataFrame()
    replay_prob_df = pd.DataFrame()
    replay_feature_df = pd.DataFrame()
    for genuine_user in tqdm(all_users):
        (
            feature_df_u,
            mather_df_u,
            replay_feature_df_u,
            replay_prob_df_u,
        ) = authentication_scenario(data.copy(), genuine_user, m, t_now)
        feature_df = pd.concat([feature_df, feature_df_u])
        mather_df = pd.concat([mather_df, mather_df_u])
        replay_prob_df = pd.concat([replay_prob_df, replay_prob_df_u])
        replay_feature_df = pd.concat([replay_feature_df, replay_feature_df_u])

    feature_df.to_csv("../result/Adaptive/Normal/feature_df.csv", index=False)
    mather_df.to_csv("../result/Adaptive/Normal/mather_df.csv", index=False)
    replay_prob_df.to_csv("../result/Adaptive/Normal/replay_prob_df.csv", index=False)
    replay_feature_df.to_csv("../result/Adaptive/Normal/replay_feature_df.csv", index=False)
