import pandas as pd
from scoring_players import scorer, exhaustion, players_in_match, player_form,opponent_team_form
from datetime import date
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import matplotlib.pyplot as pyplot
import os

# import sklearn
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

matches = os.listdir('matches')
print(matches[0])


for i, match in enumerate(matches):
    with open('matches/'+match, 'r') as file:
        # Read the content of the file line by line and create a list
        players = file.readlines()
    if len(players) == 0:
        print(0)
        continue
    #print(players[0].replace('\\', '/'))
    match_ref = match.replace('_', '/')
    data = pd.read_csv(players[0].replace('\\', '/').strip())
    #print(match_ref)
    team_1 = data.loc[data['Match Report'] == match_ref, 'Squad'].values[0]
    team_2 = data.loc[data['Match Report'] == match_ref, 'Opponent'].values[0]
    #print(team_1, team_2)
    team_1_form, team_2_form = 0, 0
    n_1_players, n_2_players = 0, 0
    for player in players:
        data = pd.read_csv(player.replace('\\', '/').strip())
        player_team = data.loc[data['Match Report'] == match_ref, 'Squad'].values[0]

        if player_team == team_1:
            n_1_players+=1
            team_1_form+=data.loc[data['Match Report'] == match_ref, 'Form'].values[0]
        elif player_team ==team_2:
            n_2_players += 1
            team_2_form += data.loc[data['Match Report'] == match_ref, 'Form'].values[0]
    team_1_form = team_1_form/n_1_players
    team_2_form = team_2_form/n_2_players
    #print(team_1_form, team_2_form)
    for player in players:
        data = pd.read_csv(player.replace('\\', '/').strip())
        player_team = data.loc[data['Match Report'] == match_ref, 'Squad'].values[0]
        if player_team == team_1:
            opponent_form = team_2_form
        elif player_team == team_2:
            opponent_form = team_1_form
        else:
            print('Player team:', player_team, 'teams:', team_1, ',', team_2)
            raise ValueError()
        if 'Opponent form' not in data:
            data['Opponent form'] = 0.5
            data.loc[data['Match Report'] == match_ref, 'Opponent form'] = opponent_form

        else:
            data.loc[data['Match Report'] == match_ref, 'Opponent form'] = opponent_form
        print(player.replace('\\', '/').strip())
        data.to_csv(player.replace('\\', '/').strip())
        #print(data['Opponent form'].values)
    print('Progress: match {0}/{1}'.format(i, len(matches)))

# players_in_match(1, 2)
path = 'csvs/Patrick-van-Aanholt/2019-2020/summary.csv'
all_players = [os.path.join(subdir, files[-1]) for subdir, dirs, files in os.walk('csvs') if 'summary.csv' in files]
print(all_players)
# for player in all_players:
#     form = player_form(pd.read_csv(player))
#     form.to_csv(player)

a=b
player_stats = pd.read_csv(path)

print(opponent_team_form('2019-2020', player_stats.tail(5)))
a=b
print(player_stats.loc[player_stats['Round'] == 'Matchweek 1']['Squad'].values[0])
a=b
player_stats = player_stats.drop(player_stats[player_stats['Pos'] == 'On matchday squad, but did not play'].index)

player_stats = player_form(player_stats)
print(player_stats['Match Report'][0])
print(len(players_in_match('2019-2020', player_stats['Match Report'][0])))
print(player_stats.head(7))

player_stats['FPL Score'] = player_stats.apply(scorer, axis=1)  # Adding new column with FPL score (no bonus point)
print('hi', player_stats.index)
print(player_stats.iloc[24 - 3]['Date'])
player_stats = exhaustion(player_stats)



print(player_stats.head(10))
#
#
# match_date = player_stats.Date[0]
#
# match_date = date(int(match_date[0:4]), int(match_date[5:7]), int(match_date[8:10]))
#
# # with_exhaustion = exhaustion(player_stats)
# # print(with_exhaustion.head())
#
# # Try to make a cube
# relevant_columns = ['Round', 'Venue', 'FPL Score']
# relevant = player_stats[player_stats.columns.intersection(relevant_columns)]
# relevant['Round'] = relevant['Round'].str[-1]
# relevant['Venue'] = relevant['Venue'].replace(r'Home', '1', regex=True)
# relevant['Venue'] = relevant['Venue'].replace(r'Away', '0', regex=True)
# relevant['Venue'] = relevant['Venue'].astype(float)
# relevant['Round'] = relevant['Round'].astype(float)
# print(relevant.head())
#
# relevant = series_to_supervised(relevant)
# values = relevant.values
# print(relevant.head())
# n_train_weeks = 15
# train = values[:n_train_weeks, :]
# test = values[n_train_weeks:, :]
#
# train_X, train_y = train[:, :-1], train[:, -1]
# test_X, test_y = test[:, :-1], test[:, -1]
#
# train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
# test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
# print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
#
# # design network
# model = Sequential()
# model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
# model.add(Dense(1))
# model.compile(loss='mae', optimizer='adam')
# # fit network
# [print(i.shape, i.dtype) for i in model.inputs]
# [print(o.shape, o.dtype) for o in model.outputs]
# [print(l.name, l.input_shape, l.dtype) for l in model.layers]
# history = model.fit(train_X, train_y, epochs=50, batch_size=1, validation_data=(test_X, test_y), verbose=2,
#                     shuffle=False)
# # plot history
# pyplot.plot(history.history['loss'], label='train')
# pyplot.plot(history.history['val_loss'], label='test')
# pyplot.legend()
# pyplot.show()
#
# year = '/2022-2023/'
# #for path in attackers:
# #    if not path.exists(path + year):
# #        continue
# #    df = pd.read_csv(path + year + 'summary.csv')
