import numpy as np
import pandas as pd
import os
from datetime import datetime
from scoring_players import scorer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from log_regression import LogisticRegressionTest

import joblib


seasons = ['2022-2023', '2023-2024']
#seasons = ['2021-2022', '2022-2023'] '2023-2024', '2019-2020',

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
fpl_scores = []
min_played = []
start_column = []
form = []
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
        # min_played.append(df["Min"].tolist())
        # start_column = df["Start"]

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
                form.append(df.loc[index - 1]["Form"])


days_interval = np.array(days_interval)
form = np.array(form)
fpl_scores = np.array(fpl_scores)
min_played = np.array(min_played)
start_column = np.array(start_column)
X = np.hstack((days_interval.reshape(-1, 1), fpl_scores.reshape(-1, 1), min_played.reshape(-1, 1), form.reshape(-1, 1)))

nan_rows = np.isnan(X).any(axis=1)
X = X[~nan_rows]

y = start_column
y = y[~nan_rows]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1234)

#clf = LogisticRegressionTest(0.02, 1000)
clf = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "LogisticRegression", "attackers.joblib"))
# clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)




accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Compute precision
precision = precision_score(y_test, y_pred)
print("Precision:", precision)

# Compute recall
recall = recall_score(y_test, y_pred)
print("Recall:", recall)

# Compute F1 score
f1 = f1_score(y_test, y_pred)
print("F1 Score:", f1)

# Compute confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# if(accuracy > 0.72):
#     joblib.dump(clf, os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "LogisticRegression", "goalkeepers.joblib"))
