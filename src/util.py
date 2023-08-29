import random


def prepare_user_configurations_updated(
    data,
    genuine_user="s002",
    feature_ratio=0.25,
    classifier_ratio=0.25,
    imposter_ratio=0.5,
):
    random.seed(42)
    users = data["subject"].unique().tolist()
    users.remove(genuine_user)
    random.shuffle(users)

    feature_users_count = int(len(users) * feature_ratio)
    classifier_users_count = int(len(users) * classifier_ratio)

    feature_users = users[:feature_users_count]
    classifier_users = users[
        feature_users_count : feature_users_count + classifier_users_count
    ]
    imposter_users = users[feature_users_count + classifier_users_count :]

    return genuine_user, feature_users, classifier_users, imposter_users
