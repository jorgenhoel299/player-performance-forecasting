import pandas as pd
from scoring_players import scorer

player_stats = pd.read_csv('csvs/Aaron-Wan-Bissaka/2023-2024.csv')
player_stats = player_stats.iloc[:-1] # Last row has a summary, which we don't currently need

player_stats['FPL Score'] = player_stats.apply(scorer, axis=1) #Adding new column with FPL score (no bonus point)

print(player_stats.head())
