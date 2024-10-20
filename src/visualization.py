import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scoring_players import scorer, exhaustion
import seaborn as sns
# setting up for nice plots
plt.rcParams['text.usetex'] = True
plt.rcParams['pgf.rcfonts'] = False
plt.rcParams['pgf.preamble'] = r'\usepackage{amsmath}'
plt.switch_backend('pgf')
plt.rcParams.update({'font.size': 12})
plt.rcParams['pgf.texsystem'] = 'pdflatex'

# either concatenate all data into one table and save, or load already created file
create_data = 0
if create_data == 1:
    exhaust = []
    form = []
    FPL_score = []
    home_away = []


    all_players = [os.path.join(subdir, files[-1]) for subdir, dirs, files in os.walk('csvs') if 'summary.csv' in files]
    for i, player in enumerate(all_players):

        data = pd.read_csv(player)
        data = exhaustion(data)
        data['FPL Score'] = data.apply(scorer, axis=1)
        data = data[data['Comp'] == 'Premier League']
        form = form + list(data['Form'].values)
        exhaust = exhaust + list(data['nr_of_matches_8_days'].values)
        FPL_score = FPL_score + list(data['FPL Score'].values)
        home_away = home_away + list(data['Venue'].values)
        print('{}/{}'.format(i, len(all_players)))

    data = pd.DataFrame({
        'Form': form,
        'Exhaustion': exhaust,
        'Venue': home_away,
        'FPL Score': FPL_score,
    })
    data.to_csv('tables/all_data.csv')
else:
    data = pd.read_csv('tables/all_data.csv')
data['Venue'] = data['Venue'].replace(r'Home', '1', regex=True)
data['Venue'] = data['Venue'].replace(r'Away', '0', regex=True)
data['Venue'] = data['Venue'].astype(float)

sns.histplot(data['FPL Score'], bins=30)#, kde=True)
print(data.values.shape)
plt.savefig('figures/FPL_score_hist.pdf', format='pdf', bbox_inches='tight')

example_player = ['csvs\Abdoulaye-Doucoure/2021-2022/summary.csv', 'csvs\Aaron-Hickey/2023-2024/summary.csv', 'csvs/Mark-Noble/2019-2020/summary.csv', 'csvs\Erling-Haaland/2022-2023/summary.csv']
example_player = [i.replace('\\', '/') for i in example_player]


fig, axs = plt.subplots(2, 2, figsize=(10, 8))
axs = axs.flatten()
for i, player in enumerate(example_player):
    data = pd.read_csv(player)
    data['FPL Score'] = data.apply(scorer, axis=1)
    FPL = data['FPL Score'].values[:10]
    x = np.linspace(1, 10, 10)

    axs[i].plot(x, FPL, label=f'Function {i+1}', marker='o')
    axs[i].set_title(f'Line Graph {i+1}')

    #axs[i].set_ylim(0, 15)

    #axs[i].set_yticks([0, 2, 4, 6, 8, 10, 12, 14])
    axs[i].set_xticks(x)
    name = player.split('/')[1].replace('-', ' ')
    axs[i].set_title(name)
    if i == 0 or i == 2:
        axs[i].set_ylabel('FPL Score')
    if i ==2 or i ==3:
        axs[i].set_xlabel('Round')
plt.tight_layout()
plt.savefig('figures/fpl.pdf', format='pdf', bbox_inches='tight')