import numpy as np
import pandas as pd
import os
from datetime import datetime
from scoring_players import scorer

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from log_regression import LogisticRegressionTest
from sklearn import datasets

seasons = ['2023-2024', '2019-2020', '2020-2021', '2021-2022', '2022-2023']


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

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "positions//midfielderrs.txt"), "r") as f:
    for line in f:
        defenders.append(line.strip())

for season in seasons:
    for defender in defenders:

        if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "csvs", defender, season,"summary.csv")):
            df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "csvs", defender, season,"summary.csv"))
        else:
            continue
        position = most_played_pos(df["Pos"])
        date_column = df["Date"]
        #min_played.append(df["Min"].tolist())
        #start_column = df["Start"]

        for i, date in enumerate(date_column):
            date = datetime.strptime(date, '%Y-%m-%d')
            if i == 0:
                days_interval.append(0)
            else:
                days_interval.append((date - datetime.strptime(date_column[i - 1], '%Y-%m-%d')).days)
        for index, row in df.iterrows():
            fpl_scores.append(scorer(row))
            min_played.append(row["Min"])
            start_column.append(1 if row["Start"] == "Y" else 0)

days_interval = np.array(days_interval)
fpl_scores = np.array(fpl_scores)
min_played = np.array(min_played)
start_column = np.array(start_column)
X = np.hstack((days_interval.reshape(-1, 1), fpl_scores.reshape(-1, 1), min_played.reshape(-1, 1)))

nan_rows = np.isnan(X).any(axis=1)
X = X[~nan_rows]

y = start_column
y = y[~nan_rows]
#---------------------------------------------------------
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# model = LogisticRegression()
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# confusion = confusion_matrix(y_test, y_pred)
# report = classification_report(y_test, y_pred)

# print("Accuracy:", accuracy)
# print("Confusion Matrix:\n", confusion)
# print("Classification Report:\n", report)

# bc = datasets.load_breast_cancer()
# X, y = bc.data, bc.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1234)

clf = LogisticRegressionTest(0.001, 5000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
def accuracy(y_pred, y_test):
    return np.sum(y_pred == y_test)/len(y_test)

print(y_pred)
print(X)

