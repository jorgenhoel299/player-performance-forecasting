import numpy as np
import pandas as pd
import os
from datetime import datetime
from scoring_players import scorer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from linear_regression import LinearRegressionOwn
from sklearn.linear_model import LinearRegression
from lsm import LSM

#seasons = ['2023-2024', '2019-2020', '2020-2021', '2021-2022', '2022-2023']
seasons = ['2019-2020', '2020-2021', '2021-2022', '2022-2023']

def plot_stuff(x1,x2,x3,x4,x5,y):
    # Create a 2x3 grid for subplots
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))

    # Scatter plot on the first subplot
    axs[0, 0].scatter(x1, y)
    axs[0, 0].set_title('Scatter Subplot 1')

    # Scatter plot on the second subplot
    axs[0, 1].scatter(x2, y)
    axs[0, 1].set_title('Scatter Subplot 2')

    # Scatter plot on the third subplot
    axs[0, 2].scatter(x3, y)
    axs[0, 2].set_title('Scatter Subplot 3')

    # Scatter plot on the fourth subplot
    axs[1, 0].scatter(x4, y)
    axs[1, 0].set_title('Scatter Subplot 4')

    # Scatter plot on the fifth subplot
    axs[1, 1].scatter(x5, y)
    axs[1, 1].set_title('Scatter Subplot 5')

    # Remove empty subplot
    fig.delaxes(axs[1, 2])

    # Adjust layout to prevent overlapping titles
    plt.tight_layout()

    # Show the plot
    plt.show()


def most_played_pos(input_list):
    frequency_dict = {}

    for element in input_list:
        if element in frequency_dict:
            frequency_dict[element] += 1
        else:
            frequency_dict[element] = 1
    most_frequent_element = max(frequency_dict, key=frequency_dict.get)

    return most_frequent_element


defenders = []
days_interval = []
min_played = []
start_column = []
form = []
opponent_form = []
fpl_scores = []
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "positions//attackers.txt"), "r") as f:
    for line in f:
        defenders.append(line.strip())

for season in seasons:
    for defender in defenders:
        if os.path.exists(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "csvs", defender, season, "summary.csv")):
            df = pd.read_csv(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "csvs", defender, season, "summary.csv"))
        else:
            continue
        position = most_played_pos(df["Pos"])
        date_column = df["Date"]

        for i, date in enumerate(date_column):
            date = datetime.strptime(date, '%Y-%m-%d')
            if i == 0:
                continue
            else:
                days_interval.append((date - datetime.strptime(date_column[i - 1], '%Y-%m-%d')).days)
        for index, row in df.iterrows():
            if index == 0:  # first game means no fpl score
                continue
            else:
                fpl_scores.append(scorer(df.loc[index - 1]))
                min_played.append(df.loc[index - 1]["Min"])
                start_column.append(0 if df.loc[index - 1]["Start"] == "N" else 1)
                form.append(df.loc[index - 1]["Form"])
                opponent_form.append(df.loc[index - 1]["Opponent form"])

def min_max_scaling(data):
    min_val = min(data)
    max_val = max(data)
    scaled_data = [(x - min_val) / (max_val - min_val) for x in data]
    return scaled_data

days_interval = min_max_scaling(days_interval)
form = min_max_scaling(form)
min_played = min_max_scaling(min_played)
opponent_form = min_max_scaling(opponent_form)


days_interval = np.array(days_interval)
form = np.array(form)
fpl_scores = np.array(fpl_scores)
min_played = np.array(min_played)
start_column = np.array(start_column)
opponent_form = np.array(opponent_form)

X = np.hstack((
    days_interval.reshape(-1, 1),
    min_played.reshape(-1, 1),
#    start_column.reshape(-1, 1),
    form.reshape(-1, 1),
    opponent_form.reshape(-1, 1)
))

# Replace NaN values with 0 for both X and y
X = np.nan_to_num(X)
y = np.nan_to_num(fpl_scores)

# Use y directly if it's a 1D array
y = y[~np.isnan(X).any(axis=1)]

# Identify rows with NaN values in X
nan_rows_X = np.isnan(X).any(axis=1)

# Remove corresponding rows from both X and y
X = X[~nan_rows_X]
y = y[~nan_rows_X]

nan_in_X = np.isnan(X).any()
nan_in_y = np.isnan(y).any()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1234)

#plot_stuff(days_interval, min_played, start_column, form, opponent_form, y)


reg = LinearRegressionOwn(lr=0.001)
reg.fit(X_train,y_train)
predictions = reg.predict(X_test)

def mse(y_test, predictions):
    return np.mean((y_test-predictions)**2)

def mean_absolute_error(y_test, predictions):
    n = len(y_test)
    mae = np.sum(np.abs(y_test - predictions)) / n
    return mae

# mse = mse(y_test, predictions)
# print(mse)
# print(mean_absolute_error(y_test, predictions))


sklearn_reg = LinearRegression()
sklearn_reg.fit(X_train, y_train)

# Predictions
custom_predictions = reg.predict(X_test)
sklearn_predictions = sklearn_reg.predict(X_test)


lsm_reg = LSM()
lsm_reg.fit(X_train, y_train)
lsm_pred = lsm_reg.predict(X_test)

# Calculate MAE for both models
custom_mae = mean_absolute_error(y_test, custom_predictions)
sklearn_mae = mean_absolute_error(y_test, sklearn_predictions)
lsm_mae = mean_absolute_error(y_test, lsm_pred)

custom_mse = mse(y_test, custom_predictions)
sklearn_mse = mse(y_test, sklearn_predictions)
lsm_mse = mse(y_test, lsm_pred)
print("Gradiend descent Linear Regression MAE:", custom_mae)
print("LSM Regression MAE:", lsm_mae)
print("Scikit-learn Linear Regression MAE:", sklearn_mae)
print("Gradiend descent Linear Regression MSE:", custom_mse)
print("LSM Regression MSE:", lsm_mse)
print("Scikit-learn Linear Regression MSE:", sklearn_mse)