import os
import pandas as pd
import numpy as np


def get_directories_in_path(path):
    directories = []
    for root, dirs, files in os.walk(path):
        for dir_name in dirs:
            directories.append(dir_name)
    return directories


def most_played_pos(input_list):
    frequency_dict = {}
    for element in input_list:
        if element in frequency_dict:
            frequency_dict[element] += 1
        else:
            frequency_dict[element] = 1
    most_frequent_element = max(
        (key for key, value in frequency_dict.items() if pd.notna(key)),
        key=frequency_dict.get,
        default=None
    )

    return most_frequent_element


def get_position(df, player):
    pos = most_played_pos(df["Pos"])

    # for i in range(len(df)):
    #     if (not pd.isna(df.iloc[i]["Pos"])) and len(df.iloc[i].Pos) < 6:
    #         pos = df.iloc[i].Pos
    #         break

    if "," in pos:
        pos = pos.split(",")[0]

    #    print(pos)
    if pos in ['RB', 'CB', 'LB']:
        return 'Defender'
    elif pos in ['CM', 'DM', 'AM', 'RM', 'LM']:
        return 'Midfielder'
    elif pos in ['RW', 'LW', 'FW']:
        return 'Forward'
    elif pos == 'GK':
        return 'Goalkeeper'
    else:
        # raise ValueError('unknown position {}. consider adding to classify player.'.format(pos))
        return pos


seasons = ['2023-2024', '2019-2020', '2020-2021', '2021-2022', '2022-2023']
button_links = ['summary', 'passing', 'passing_types', 'gca', 'defense', 'possession', 'misc']

df = pd.DataFrame()

folder_path = os.path.dirname(os.path.abspath(__file__)) + "\\csvs\\"
player_names = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]

already_parsed_player = []

goalkeepers = []
defenders = []
midfielderrs = []
attackers = []

seasons = ["2019-2020", "2020-2021", "2021-2022", "2022-2023", "2023-2024"]

for season in seasons:
    for player in player_names:
        if player in already_parsed_player:
            continue
        # print(player)
        season_path = os.path.join(folder_path, player, season)
        if os.path.isdir(season_path):
            files = [os.path.join(season_path, f) for f in os.listdir(season_path) if
                     os.path.isfile(os.path.join(season_path, f))]
            file = files[0]
            df = pd.read_csv(file)

            position = get_position(df, player)
            # print(player)
            if "Takehiro-Tomiyasu" in player:
                print(position)
            if (position == " "):
                continue

            if position == "Defender":
                defenders.append(player)
            elif position == "Midfielder":
                midfielderrs.append(player)
            elif position == "Forward":
                attackers.append(player)
            elif position == "Goalkeeper":
                goalkeepers.append(player)
            already_parsed_player.append(player)
            # else:
            #     print(df)

if not os.path.isdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "positions")):
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "positions"))

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "positions//defenders.txt"), "w") as f:
    f.write("\n".join(defenders))
    f.close()
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "positions//midfielderrs.txt"), "w") as f:
    f.write("\n".join(midfielderrs))
    f.close()
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "positions//attackers.txt"), "w") as f:
    f.write("\n".join(attackers))
    f.close()
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "positions//goalkeepers.txt"), "w") as f:
    f.write("\n".join(goalkeepers))
    f.close()
# path_to_search = os.path.dirname(os.path.abspath(__file__)) + "\\csvs\\"
# player_directories = get_directories_in_path(path_to_search)
#
# files = []
# for player in player_directories:
#     files = [os.path.join(path_to_search, player, file) for file in os.listdir(os.path.join(path_to_search, player))]
#     print(player, files)
#     break
#
# file = files[0]
# df = pd.read_csv(file)
# columns = df.columns
