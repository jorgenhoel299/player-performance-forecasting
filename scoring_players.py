import math
import os
import numpy as np

from datetime import date
from datetime import timedelta

import pandas as pd


def scorer(row):
    print(row)
    playing_time = row.Min
    goals_scored = row.Gls
    assists = row.Ast
    penalties_missed = row.PKatt
    goals_conceded = int(row.Result[4])
    yellow_cards = row.CrdY
    red_cards = row.CrdR
    own_goal = 0  # TODO

    if row.Pos in ['RB', 'CB', 'LB']:
        position = 'Defender'
    elif row.Pos in ['CM', 'DM', 'AM', 'RM', 'LM']:
        position = 'Midfielder'
    elif row.Pos in ['FW', 'RW', 'LW']:
        position = 'Forward'
    elif row.Pos == 'GK':
        position = 'Goalkeeper'
    else:
        raise ValueError('unknown position{}. consider adding to classify player.'.format(row.Pos))

    score = 0
    score += assists * 3
    score -= penalties_missed * 2
    score = score - math.floor(goals_conceded / 2) if position == 'Goalkeeper' or position == 'Defender' else score
    score -= yellow_cards
    score -= red_cards * 3
    score -= own_goal * 2

    if playing_time > 0:
        score += 1
        if playing_time > 60:
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
                matches_played = matches_played + 1 if threshold_date <= make_date(df.Date[index - i]) <= match_date else matches_played
        df.nr_of_matches_8_days[index] = matches_played
    return df


def bonus_points(player):
    # TODO make a more sophisticated way to calculate bnp
    return 5

def players_in_match(season, match_ref):
    # Returns csvs of all players in a given match in a season
    # TODO: test
    players = []
    for subdir, dirs, files in os.walk("csvs/Teemu-Pukki"):
        for file in files:
            print(os.path.join(subdir, file))
            df = pd.read_csv(os.path.join(subdir, file))
            if '07311954' in df['Match Report'].values:
                players.append(os.path.join(subdir, file))
    return players


def bonus_points_distributor(all_players):
    # Attributes bonus points to ech player in match. total sums to 6
    total_points = sum(all_players)
    #partial_bnp = player/total_points * 6
    return
    #etc