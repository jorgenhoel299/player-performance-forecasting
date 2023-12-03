import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from scoring_players import scorer
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import os

pd.options.mode.chained_assignment = None  # default='warn'
attackers = []
days_interval = []
fpl_scores = []
min_played = []
start_column = []


with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "positions//attackers.txt"), "r") as f:
    for line in f:
        attackers.append(line.strip())

seasons = ['2019-2020', '2020-2021', '2021-2022', '2022-2023']
min_rounds = 20
data = np.zeros((2, 1, min_rounds))
relevant_columns = ['Round', 'Venue', 'FPL Score']

y = np.zeros((1, min_rounds))

first = 1
# preparing data and labels
for j, season in enumerate(seasons):
    for i, attacker in enumerate(attackers):

        if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "csvs", attacker, season,"summary.csv")):
            df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "csvs", attacker, season,"summary.csv"))
        else:
            continue

        df_premier = df[df['Comp'] == 'Premier League']
        if df_premier.shape[0] < min_rounds:
            continue

        date_column = df["Date"]
        df_premier['FPL Score'] = df_premier.apply(scorer, axis=1)

        relevant_data = df_premier[df_premier.columns.intersection(relevant_columns)].head(min_rounds)
        relevant_data['Round'] = relevant_data['Round'].str[-1]
        relevant_data['Venue'] = relevant_data['Venue'].replace(r'Home', '1', regex=True)
        relevant_data['Venue'] = relevant_data['Venue'].replace(r'Away', '0', regex=True)
        relevant_data['Venue'] = relevant_data['Venue'].astype(float)
        relevant_data = relevant_data[relevant_data.Round != 'e']
        relevant_data = relevant_data[relevant_data.Round != 'd']
        relevant_data['Round'] = relevant_data['Round']

        # removing rounds that ends in character, as they are not the Premier League
        mask = ~relevant_data['Round'].apply(lambda x: any(char.isalpha() for char in x))
        relevant_data = relevant_data[mask]
        player_season_data = relevant_data[relevant_data.columns.intersection(['Round', 'Venue'])].values.astype(float).T

        if first == 1:
            data[:, 0, :] += player_season_data.astype(float)
            y[0, :] += relevant_data['FPL Score'].values
            first = 0
        else:
            data = np.concatenate([data, player_season_data[:, np.newaxis, :]], axis=1)
            y = np.vstack((y, relevant_data['FPL Score'].values))

data = np.transpose(data, (1, 2, 0))  # Converted to keras expected input shape
print(data.shape, y.shape)

all_indices = list(range(data.shape[0]))
train_ind, test_ind = train_test_split(all_indices, test_size=0.2)
print(len(train_ind), len(test_ind))
x_train, x_test = data[train_ind,:, :], data[test_ind, :, :]
y_train, y_test = y[train_ind,:], y[test_ind, :]

# Build model
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(data.shape[1], data.shape[2])))
model.add(Dense(1))  # Output layer for regression

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
print(x_train.shape, y_train.shape)

history = model.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate the model on test data
loss, mae = model.evaluate(x_test, y_test)
print(f'Mean Absolute Error on Test Data: {mae}')
# Plot training and validation loss over epochs
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
