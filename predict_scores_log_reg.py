import joblib
import numpy as np
import pandas as pd
import os
from datetime import datetime
from scoring_players import scorer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from log_regression import LogisticRegressionTest


def most_played_pos(input_list):
    frequency_dict = {}

    for element in input_list:
        if element in frequency_dict:
            frequency_dict[element] += 1
        else:
            frequency_dict[element] = 1
    most_frequent_element = max(frequency_dict, key=frequency_dict.get)

    return most_frequent_element


players = []
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "positions//defenders.txt"), "r") as f:
    for line in f:
        players.append(line.strip())

seasons = ["2023-2024"]
players_with_data = {}
players_actually_started = {}
for season in seasons:
    for player in players:
        days_interval = []
        form = []
        min_played = []
        start_column = []
        fpl_scores = []
        if os.path.exists(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "csvs", player, season, "summary.csv")):
            df = pd.read_csv(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "csvs", player, season, "summary.csv"))
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

        days_interval = np.array(days_interval)
        form = np.array(form)
        fpl_scores = np.array(fpl_scores)
        min_played = np.array(min_played)
        start_column = np.array(start_column)
        players_actually_started[player] = start_column.reshape(-1, 1)[-5:]
        # last five games
        players_with_data[player] = np.hstack((days_interval.reshape(-1, 1)[-5:], fpl_scores.reshape(-1, 1)[-5:],
                                               min_played.reshape(-1, 1)[-5:], form.reshape(-1, 1)[-5:]))

model = joblib.load(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "LogisticRegression", "attackers.joblib"))
players_with_chances_to_start = {}

for player in players_with_data:
    predictions = model.predict(players_with_data[player])
    if len(predictions) == 0:
        players_with_chances_to_start[player] = np.array([0])
        continue
    players_with_chances_to_start[player] = predictions.count(1) / len(predictions)

sorted_players = sorted(players_with_chances_to_start.items(), key=lambda item: item[1], reverse=True)
acc = 0
# Print the first 5 players based on their values
print(players_with_data["Ben-Godfrey"])
for player, perc in sorted_players:
    start_values = players_actually_started[player]
    if len(start_values) < 5:
        continue
    actual_perc = np.count_nonzero(start_values == 1) / len(start_values)
    print(f"{player}: {perc}, actual: {actual_perc}")
    acc += 1 if abs(perc -actual_perc) <=0.2 else 0
print("Accuracy last 5 games:", acc/len(sorted_players))
