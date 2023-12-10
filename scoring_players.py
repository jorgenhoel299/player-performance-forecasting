import math
import os
import numpy as np

from datetime import date
from datetime import timedelta

import pandas as pd


def scorer(row, position=""):
    playing_time = row.Min
    goals_scored = row.Gls
    assists = row.Ast
    penalties_missed = row.PKatt
    result = row.Result
    goals_conceded = int(row.Result.split("–")[1][0])
    yellow_cards = row.CrdY
    red_cards = row.CrdR
    own_goal = 0  # TODO

    # if position == "":
    pos = row.Pos
    if pd.isna(row["Pos"]):
        return 0
    for p in pos.split(","):
        if p in ['RB', 'CB', 'LB', 'RB,LB', 'LB,WB', 'RB,CB', 'WB']:
            position = 'Defender'
            break
        elif p in ['CM', 'DM', 'AM', 'RM', 'LM']:
            position = 'Midfielder'
            break
        elif p in ['FW', 'RW', 'LW', 'AM,LW']:
            position = 'Forward'
            break
        elif p == 'GK':
            position = 'Goalkeeper'
            break
        else:
            continue
        # if pos == "":
        #     raise ValueError("pos not defined:", pos)pos

    score = 0
    score += assists * 3
    score -= penalties_missed * 2
    score = score - math.floor(goals_conceded / 2) if position == 'Goalkeeper' or position == 'Defender' else score
    score -= yellow_cards
    score -= red_cards * 3
    score -= own_goal * 2

    if int(playing_time) > 0:
        score += 1
        if int(playing_time) > 60:
            score += 1

    if goals_scored > 0:
        if position == 'Forward':
            score += goals_scored * 4
        elif position == 'Midfielder':
            score += goals_scored * 5
        elif position == 'Defender' or position == 'Goalkeeper':
            score += goals_scored * 6
        else:
            raise ValueError

    if goals_conceded == 0:
        if position == 'Midfielder':
            score += 1
        elif position == 'Defender' or position == 'Goalkeeper':
            score += 4

    if position == 'Goalkeeper':
        return 1
        score += math.floor(shots_saved / 3)
        score += penalties_saved * 5

    return score


def make_date(match_date):
    return date(int(match_date[0:4]), int(match_date[5:7]), int(match_date[8:10]))


def exhaustion(df):
    # adds a column with the number of matches where the player played more then 30 minutes
    # Checks 3 most recent matches, highest value possible is 3, lowest 0
    # TODO: test that it works propertly
    df['nr_of_matches_8_days'] = 0
    for index, row in df.iterrows():

        if index == 0:
            continue

        # print(index, row)
        match_date = make_date(df.Date[index])
        threshold_date = match_date - timedelta(days=8)

        matches_played = 0
        if index == 1:
            if threshold_date <= make_date(df.Date[0]) <= match_date:
                matches_played += 1
        elif index == 2:
            if threshold_date <= make_date(df.Date[0]) <= match_date:
                matches_played += 1
            if threshold_date <= make_date(df.Date[1]) <= match_date:
                matches_played += 1
        else:
            for i in range(1, 4):
                matches_played = matches_played + 1 if threshold_date <= make_date(
                    df.iloc[index - i]['Date']) <= match_date else matches_played
        df.loc[index, ('nr_of_matches_8_days')] = matches_played
    return df


def player_form(player):
    if 'Form' in player.columns:
        return player
    # ratio of matches won 5 last games
    player = results_as_floats(player)
    player['Form'] = player['Result_num'].shift(1).rolling(window=3).mean()
    player.loc[:2, 'Form'] = 0.5
    return player




def opponent_form():
    matches = os.listdir('matches')
    for i, match in enumerate(matches):
        with open('matches/'+match, 'r') as file:
            # Read the content of the file line by line and create a list
            players = file.readlines()
        if len(players) == 0:
            continue
        match_ref = match.replace('_', '/')
        data = pd.read_csv(players[0].replace('\\', '/').strip())
        team_1 = data.loc[data['Match Report'] == match_ref, 'Squad'].values[0]
        team_2 = data.loc[data['Match Report'] == match_ref, 'Opponent'].values[0]
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
        print('Progress: match {0}/{1}'.format(i, len(matches)))

def results_as_floats(player):
    player['Result_num'] = player['Result']
    player['Result_num'] = player['Result_num'].replace(to_replace='.*W.*', value=1,
                                                        regex=True)  # .astype(float)
    player['Result_num'] = player['Result_num'].replace(to_replace='.*D.*', value=0.5,
                                                        regex=True)  # .astype(float)
    player['Result_num'] = player['Result_num'].replace(to_replace='.*L.*', value=0,
                                                        regex=True)  # .astype(float)
    player['Result_num'] = player['Result_num'].astype(float)
    return player


def players_in_match(season, match_ref):
    # Returns csvs of all players in a given match in a season
    # TODO: test
    all_players = []
    for subdir, dirs, files in os.walk("csvs"):
        if season not in subdir or 'html' in subdir:
            continue
        for file in files:
            if 'summary' not in file:
                continue
            df = pd.read_csv(os.path.join(subdir, file))
            if match_ref in df['Match Report'].values:
                all_players.append(os.path.join(subdir, file))
    return all_players
