import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from scoring_players import scorer, exhaustion
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

relevant_columns = ['Venue', 'nr_of_matches_8_days', 'Form', 'Opponent form', 'FPL Score']
data = np.zeros((len(relevant_columns)-1, 1, min_rounds))
y = np.zeros((1, min_rounds))

first = 1
# preparing data and labels
for j, season in enumerate(seasons):
    for i, attacker in enumerate(attackers):

        if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "csvs", attacker, season,"summary.csv")):
            df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "csvs", attacker, season,"summary.csv"))
        else:
            continue
        df = exhaustion(df)
        df_premier = df[df['Comp'] == 'Premier League']  # filtering out non-PL games
        if df_premier.shape[0] < min_rounds:
            continue

        df_premier['FPL Score'] = df_premier.apply(scorer, axis=1)
        relevant_data = df_premier[df_premier.columns.intersection(relevant_columns)].head(min_rounds)
        if 'Round' in relevant_columns:
            relevant_data = relevant_data[relevant_data.Round != 'e']
            relevant_data = relevant_data[relevant_data.Round != 'd']
            relevant_data['Round'] = relevant_data['Round'].str[-1]

        relevant_data['Venue'] = relevant_data['Venue'].replace(r'Home', '1', regex=True)
        relevant_data['Venue'] = relevant_data['Venue'].replace(r'Away', '0', regex=True)
        relevant_data['Venue'] = relevant_data['Venue'].astype(float)

        #mask = ~relevant_data['Round'].apply(lambda x: any(char.isalpha() for char in x))
        # removing rounds that ends in character, as they are not the Premier League

        #relevant_data = relevant_data[mask
        player_season_data = relevant_data[relevant_data.columns.intersection(relevant_columns[:-1])].values.astype(float).T

        if first == 1:
            data[:, 0, :] += player_season_data.astype(float)
            y[0, :] += relevant_data['FPL Score'].values
            first = 0
        else:
            data = np.concatenate([data, player_season_data[:, np.newaxis, :]], axis=1)
            y = np.vstack((y, relevant_data['FPL Score'].values))

data = np.transpose(data, (1, 2, 0))  # Converted to keras expected input shape
print(data.shape, y.shape)

# Prepere data for training and testing
all_indices = list(range(data.shape[0]))
train_ind, test_ind = train_test_split(all_indices, test_size=0.2)
print(len(train_ind), len(test_ind))
x_train, x_test = data[train_ind,:, :], data[test_ind, :, :]
y_train, y_test = y[train_ind,:], y[test_ind, :]

# Build models

# 1 layer
min_loss = pd.DataFrame(columns=['cells_first_layer', 'train_loss'])
for n_cells_1 in [1]+[i for i in range(int(1), int(64) + 1) if i % 4 == 0]:
    print(n_cells_1)
    model = Sequential()
    model.add(LSTM(n_cells_1, activation='tanh', input_shape=(data.shape[1], data.shape[2])))
    model.add(Dense(1))  # Output layer for regression
    model.compile(optimizer='sgd', loss='mse', metrics=['mae'])
    history = model.fit(x_train, y_train, epochs=15, batch_size=32, validation_split=0.2)
    loss, mae = model.evaluate(x_test, y_test)
    min_val_loss_epoch = np.argmin(history.history['val_loss'])
    new_row = {'cells_first_layer': n_cells_1, 'train_loss': round(mae, 3)}
    min_loss = min_loss._append(new_row, ignore_index=True)

min_loss.to_csv('tables/1_layer_20.csv')
a=b
#2 layer
min_loss = pd.DataFrame(columns=['cells_first_layer', 'cells_second_layer' , 'train_loss'])
for n_cells_1 in [1]+[i for i in range(int(1), int(64) + 1) if i % 8 == 0]:
    for n_cells_2 in [1]+[i for i in range(int(1), int(64) + 1) if i % 8 == 0]:

        model = Sequential()
        model.add(LSTM(n_cells_1, return_sequences=True, activation='tanh', input_shape=(data.shape[1], data.shape[2])))
        model.add(LSTM(units=n_cells_2))
        model.add(Dense(1))  # Output layer for regression
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        history = model.fit(x_train, y_train, epochs=15, batch_size=32, validation_split=0.2)
        loss, mae = model.evaluate(x_test, y_test)
        min_val_loss_epoch = np.argmin(history.history['val_loss'])
        new_row = {'cells_first_layer': n_cells_1, 'cells_second_layer': n_cells_2, 'train_loss': round(mae, 3)}
        min_loss = min_loss._append(new_row, ignore_index=True)
min_loss.to_csv('tables/2_layer_5.csv')

# 3 layer
min_loss = pd.DataFrame(columns=['cells_first_layer', 'cells_second_layer' , 'cells_third_layer', 'train_loss'])
for n_cells_1 in [1]+[i for i in range(int(1), int(64) + 1) if i % 16 == 0]:
    for n_cells_2 in [1]+[i for i in range(int(1), int(64) + 1) if i % 16 == 0]:
        for n_cells_3 in [1] + [i for i in range(int(1), int(64) + 1) if i % 16 == 0]:

            model = Sequential()
            model.add(LSTM(n_cells_1, return_sequences=True, activation='tanh', input_shape=(data.shape[1], data.shape[2])))
            model.add(LSTM(units=n_cells_2, return_sequences=True))
            model.add(LSTM(units=n_cells_3))
            model.add(Dense(1))  # Output layer for regression
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            history = model.fit(x_train, y_train, epochs=15, batch_size=32, validation_split=0.2)
            loss, mae = model.evaluate(x_test, y_test)
            min_val_loss_epoch = np.argmin(history.history['val_loss'])
            new_row = {'cells_first_layer': n_cells_1, 'cells_second_layer': n_cells_2, 'cells_third_layer': n_cells_3, 'train_loss': round(mae, 3)}
            min_loss = min_loss._append(new_row, ignore_index=True)
min_loss.to_csv('tables/3_layer_5.csv')
# Evaluate the model on test data
loss, mae = model.evaluate(x_test, y_test)
print(f'Mean Absolute Error on Test Data: {mae}')
# Plot training and validation loss over epochs
plt.plot(history.history['loss'], label ='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
