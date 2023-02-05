import math
import pandas as pd
import numpy as np

# Load the data to a pandas dataframe
dataframe = pd.read_csv("dataset/dataset.csv")

# get the number of players
nb_users = len(dataframe.index)

# general info on the dataframe
print("---\ngeneral info on the dataframe")
print(dataframe.info())

# print the columns of the dataframe
print("---\ncolumns of the dataset")
print(dataframe.columns)

# print the first 10 lines of the dataframe
print("---\nfirst lines")
print(dataframe.head(10))

# print the correlation matrix of the dataset
print("---\nCorrelation matrix")
print(dataframe.corr())

# print the standard deviation
print("---\nStandard deviation")
print(dataframe.std())


def compute_dissimilarity(user_1_id, user_2_id):
    """
    Compute  dissimilarity betwwen two user
    based on their id.
    The meal is not a quantitative attribute.
    It is called a categorical variable.
    We must handle it differently than quantitative
    attributes.
    """
    age_weight = 3
    height_weight = 1
    job_weight = 2
    city_weight = 1
    music_style_weight = 3

    user_1_age = dataframe.loc[user_1_id][1]
    user_2_age = dataframe.loc[user_2_id][1]

    user_1_height = dataframe.loc[user_1_id][2]
    user_2_height = dataframe.loc[user_2_id][2]

    user_1_job = dataframe.loc[user_1_id][3]
    user_2_job = dataframe.loc[user_2_id][3]

    user_1_city = dataframe.loc[user_1_id][4]
    user_2_city = dataframe.loc[user_2_id][4]

    user_1_music = dataframe.loc[user_1_id][5]
    user_2_music = dataframe.loc[user_2_id][5]

    if user_1_job == user_2_job:
        dissimilarity_job = 0
    else:
        dissimilarity_job = job_weight

    if user_1_city == user_2_city:
        dissimilarity_city = 0
    else:
        dissimilarity_city = city_weight

    if user_1_music == user_2_music:
        dissimilarity_music = 0
    else:
        dissimilarity_music = music_style_weight

    # we build a hybrid dissimilarity
    dissimilarity = math.sqrt(
        (user_1_age - user_2_age) ** 2 * age_weight
        + (user_1_height - user_2_height) ** 2 * height_weight
        + dissimilarity_job
        + dissimilarity_city
        + dissimilarity_music
    )

    # print("----")
    # user_1_id = dataframe.loc[user_1_id][0]
    # user_2_id = dataframe.loc[user_2_id][0]
    # print(
    #     f"plyr 1 {user_1_id}, plyr 2 {user_2_id}, dissimilarity: {dissimilarity}"
    # )
    return dissimilarity

# build a dissimilarity matrix
dissimilarity_matrix = np.zeros((nb_users, nb_users))
print("compute dissimilarities")
for player_1_id in range(nb_users):
    for player_2_id in range(nb_users):
        dissimilarity = compute_dissimilarity(player_1_id, player_2_id)
        dissimilarity_matrix[player_1_id, player_2_id] = dissimilarity
        dissimilarity_matrix[player_2_id, player_1_id] = dissimilarity_matrix[player_1_id, player_2_id]

print("dissimilarity matrix")
print(dissimilarity_matrix)

mean_dissimilarity = np.mean(dissimilarity_matrix)
std_dissimilarity = np.std(dissimilarity_matrix)

print(f"mean dissimilarity: {mean_dissimilarity}")
print(f"std dissimilarity: {std_dissimilarity}")

np.save("dissimilarities.npy", dissimilarity_matrix)