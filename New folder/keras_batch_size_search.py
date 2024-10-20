import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from scoring_players import scorer, exhaustion
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import os

# setting up for nice plots
plt.rcParams['text.usetex'] = True
plt.rcParams['pgf.rcfonts'] = False
plt.rcParams['pgf.preamble'] = r'\usepackage{amsmath}'
plt.switch_backend('pgf')
plt.rcParams.update({'font.size': 12})
plt.rcParams['pgf.texsystem'] = 'pdflatex'

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
min_rounds = 5

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
batch_sizes = [1, 8, 16, 32, 64, 96, 128, 192, 256]
min_loss = pd.DataFrame(columns=['batch_size', 'test_loss', 'min_val_loss'])
for batch_size in batch_sizes:
    model = Sequential()
    model.add(LSTM(4, activation='tanh', input_shape=(data.shape[1], data.shape[2])))
    model.add(Dense(1))  # Output layer for regression
    model.compile(optimizer='sgd', loss='mse', metrics=['mae'])
    history = model.fit(x_train, y_train, epochs=15, batch_size=batch_size, validation_split=0.2)
    loss, mae = model.evaluate(x_test, y_test)
    min_val_loss_epoch = np.argmin(history.history['val_mae'])
    min_val_loss = history.history['val_mae'][min_val_loss_epoch]
    new_row = {'batch_size': batch_size, 'test_loss': round(mae, 3), 'min_val_loss': round(min_val_loss, 3)}
    min_loss = min_loss._append(new_row, ignore_index=True)

min_loss.to_csv('tables/batches_{0}_.csv'.format(min_rounds))

plt.plot(min_loss['batch_size'], min_loss['min_val_loss'], label='Validation Loss', marker='o')
plt.plot(min_loss['batch_size'], min_loss['test_loss'], label='Test Loss', marker='o')

plt.title('Model Validation and Test Loss Over Batch size, T(x) = {}'.format(min_rounds))
#plt.ylim((1, 2.5))
plt.xticks(batch_sizes)
plt.xlabel('Batch size')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig('figures/batch_size_comparison_lstm_5.pdf', format='pdf', bbox_inches='tight')