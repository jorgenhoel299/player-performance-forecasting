import pandas as pd
from scoring_players import scorer, exhaustion, players_in_match
from datetime import date
import numpy as np
players_in_match(1, 2)
path = '2023-2024.csv'
player_stats = pd.read_csv(path)
player_stats = player_stats.iloc[:-1] # Last row has a summary, which we don't currently need

player_stats['FPL Score'] = player_stats.apply(scorer, axis=1) #Adding new column with FPL score (no bonus point)

print(player_stats.head())
print(np.nan+1)
match_date = player_stats.Date[0]
print(int(match_date[0:4]))
match_date = date(int(match_date[0:4]), int(match_date[5:7]), int(match_date[8:10]))

with_exhaustion = exhaustion(player_stats)
print(with_exhaustion.head())

