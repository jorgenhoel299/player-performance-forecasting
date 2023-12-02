import os

import pandas as pd

from scoring_players import players_in_match

import matplotlib.pyplot as plt

# richie = pd.read_csv('csvs/Richarlison/2021-2022/summary.csv')
# print(richie['Round'].values)
# a=b
all_summaries = [os.path.join(subdir, files[-1]) for subdir, dirs, files in os.walk('csvs') if 'summary.csv' in files]

all_matches = []

for summary in all_summaries:
    summary = pd.read_csv(summary)
    matches = list(summary.loc[summary['Comp'] == 'Premier League', 'Match Report'].values)
    all_matches = all_matches + matches

print(len(all_matches))
all_matches = list(set(all_matches))
print(len(all_matches))
nr_players = []
for i, match in enumerate(all_matches):
    if 'Russian' in match or 'Ukrain' in match:
        continue
    season = match.split('-Premier-League')[0][-4:]

    print(match, season)
    season = season +'-'+ str(int(season)+1)
    players = players_in_match(season, match)

    pathname = 'matches/' + match.replace('/', '_')
    with open(pathname, 'w') as outfile:
        outfile.write('\n'.join(str(i) for i in players))

    print(i, 'out of', len(all_matches))
    nr_players.append(len(players))


for i in range(0, 15):
    print(i, nr_players.count(i))

plt.figure()
plt.hist(nr_players)
plt.show()
