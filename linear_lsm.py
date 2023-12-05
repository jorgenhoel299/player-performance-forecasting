import numpy as np
import pandas as pd
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from scoring_players import scorer
import matplotlib.pyplot as plt
import plotly.express as px
from linear_regression import LinearRegression
from lsm import LSM

seasons = ['2019-2020', '2020-2021', '2021-2022', '2022-2023', '2023-2024']
#seasons = ['2022-2023']

def min_max_scaling(data):
    min_val = min(data)
    max_val = max(data)
    scaled_data = [(x - min_val) / (max_val - min_val) for x in data]
    return scaled_data


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
mean_fpl_score = {}
calculated_score = {}
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "positions//attackers.txt"), "r") as f:
    for line in f:
        defenders.append(line.strip())

for season in seasons:
    for defender in defenders:
        score = []
        days_interval = []
        fpl_scores = []
        min_played = []
        start_column = []
        form = []
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
                start_column.append(0 if row["Start"] == "N" else 1)
                form.append(row["Form"])

        # print(defender, len(start_column))
        mean_fpl_score[defender + "-" + season] = np.mean(fpl_scores)
        calculated_score[defender + "-" + season] = (np.mean(min_played) * 0.5 + (len(min_played) + 1) * 0.3 + (
                1 + start_column.count(1)) / (len(start_column) + 1) * 0.2) / 10
dataX = []
dataY = []
season_defenders = []

seasons_to_use = []
for season_to_use in seasons:
    for defender in defenders:
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csvs", defender, season_to_use, "summary.csv")
        # Check if the CSV file exists
        if os.path.exists(csv_path):
            seasons_to_use.append(season_to_use)
            # Assuming mean_fpl_score and calculated_score are dictionaries with defender-year keys
            season_defenders.append(defender)
            fpl_key = defender + "-" + season_to_use
            calculated_score_key = defender + "-" + season_to_use

            # Check if keys exist in the dictionaries
            if fpl_key in mean_fpl_score and calculated_score_key in calculated_score:
                dataY.append(mean_fpl_score[fpl_key])
                dataX.append(calculated_score[calculated_score_key])



dataX = min_max_scaling(dataX)
dataY = min_max_scaling(dataY)
dataX = np.array(dataX)
dataY = np.array(dataY)

dataX = dataX.reshape(-1, 1)
dataY = dataY.reshape(-1, 1)

dataX[np.isnan(dataX)] = 0
dataY[np.isnan(dataY)] = 0
X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, test_size=0.2, random_state=1234)

fig = plt.figure(figsize=(8,6))
plt.scatter(dataX, dataY, color = "b", marker = "o", s = 30)
plt.show()

reg = LinearRegression(lr=0.001)
reg.fit(X_train,y_train)
predictions = reg.predict(X_test)

def mse(y_test, predictions):
    return np.mean((y_test-predictions)**2)

mse = mse(y_test, predictions)
print(mse)

y_pred_line = reg.predict(dataX)
cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(8,6))
m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
m2 = plt.scatter(X_test, y_test,  color='black', s=10)
plt.plot(dataX, y_pred_line, color='black', linewidth=2, label='Prediction')
plt.show()


lsm = LSM()
lsm.fit(X_train,y_train)
predictions = lsm.predict(X_test)

def mse(y_test, predictions):
    return np.mean((y_test-predictions)**2)

mse = mse(y_test, predictions)
print(mse)

y_pred_line = lsm.predict(dataX)
cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(8,6))
m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
m2 = plt.scatter(X_test, y_test,  color='black', s=10)
plt.plot(dataX, y_pred_line, color='black', linewidth=2, label='Prediction')
plt.show()

