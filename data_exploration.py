import pandas as pd
from scoring_players import scorer, exhaustion, players_in_match, player_form
from datetime import date
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import matplotlib.pyplot as pyplot


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


# players_in_match(1, 2)
path = 'Patrick-van-Aanholt/2019-2020/summary.csv'
player_stats = pd.read_csv(path)
player_stats['Result'].replace('', np.nan, inplace=True)  # removing empty rows
player_stats.dropna(subset=['Result'], inplace=True)

player_stats = player_stats.drop(player_stats[player_stats['Pos'] == 'On matchday squad, but did not play'].index)

player_stats = player_form(player_stats)
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
