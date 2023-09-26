import math


def fpl_score(position, playing_time, goals_scored, assists, shots_saved, penalties_saved, penalties_missed,
              goals_conceded, yellow_cards, red_cards, own_goal):
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
