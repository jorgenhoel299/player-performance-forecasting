import os
import string

import requests, re
from bs4 import BeautifulSoup, Comment
import pandas as pd
import time


#Getting data from all players in recent seasons
#Idea is to iterate over a list of seasons, and extract each player's match log for that season
#This data is stored in list of dictionairies right now, but I think we should just write
#the data frames to .csv. The column headers are not added to the data frame, maybe we can add them before
#saving to csv. Then we can work on data pre-preprocessing like eliminating rows and columns in a seperate file.

players_and_stats = [] # list with dictionairies of name and dataframe
players_in_this_season = []
seasons = ['2023-2024','2019-2020','2020-2021','2021-2022','2022-2023'] # Add more sesons to list when we are ready, start with the last season
                                                                        #bcs if the players are not anymore in epl or they retired we should not take them into account

table_columns = []
extracted_table_columns = 0
button_links = ['summary', 'passing', 'passing_types', 'gca', 'defense', 'possession', 'misc']
#premier league - c9 after the year
split_by_slashes = []
for season in seasons:
    URL_season = 'https://fbref.com/en/comps/9/' + season + '/stats/' + season +'-Premier-League-Stats'
    page = requests.get(URL_season).content
    soup = BeautifulSoup(page, "html.parser")

    # Until you scroll down on the page the tables are commented in the html, so needs to be extracted like this
    comments = soup.find_all(string=lambda text: isinstance(text, Comment))

    tables = []
#     for each in comments:
#         if 'table' in each:
#             try:
#                 tables.append(pd.read_html(each)[0])
#             except:
#                 continue
# #    data = tables[0]

    # ELEGANTLY getting all the links that are related to the players in the table.
    links = []
    for each in comments:
        if 'table' in each:
            links.append(re.findall(r'(en/players/[^\s]+)', str(each)))
    # (There are two for each player, one to the player profile, and one that shows match log for given season)
    # We only really need the match logs i guess, maybe the player-cards could be useful later.
    match_logs = links[0][1::2]
    player_cards = links[0][0::2]
    cleaned_match_logs = [link.split('"')[0] for link in match_logs] # Getting just the link

    # iterate over each player in the given season
    for i, player in enumerate(player_cards): # when getting all players remove ':5'
        print('season:{0}, progress in season:{1}/{2}'.format(season, i+1, len(player_cards)))
        name = player.split("\"")[0].split('/')[-1]
        for button in button_links:

            split_by_slashes = cleaned_match_logs[i].split("/")
            split_by_slashes[-2] = "c9"
            split_by_slashes.insert(len(split_by_slashes) - 1, button)


            URL_player = 'https://fbref.com/' + "/".join(split_by_slashes)
           # print(URL_player)
            data_player = requests.get(URL_player).text
            soup_player = BeautifulSoup(data_player, 'html.parser')
            # On these urls, the table is simpler to extract
            table_player = soup_player.find('table', class_='min_width sortable stats_table min_width shade_zero')
            # ELEGANTLY making table into array and then into a df
            data = []

            thead = table_player.find("thead")
            column_name_section = thead.find("tr", attrs={'class': None})
            rows = column_name_section.find_all("th")
            print(rows)
            for row in rows:
                table_columns.append(row.text)

            tbody = table_player.find("tbody")
            for row in tbody.find_all('tr'):
                date = tbody.find('th').text
                row_data = [date]
                for cell in row.find_all('td'):
                    if cell.text not in table_columns or "Match Report" in cell.text: #skip same columns from multiple tables in the page
                        if 'Match Report' in str(cell.text):
                            row_data.append(cell.find('a').get('href'))
                        else:
                            row_data.append(cell.text)

                data.append(row_data)
            df = pd.DataFrame(data)
            df = df.iloc[2:]
            df = df.reset_index(drop=True)
            df.columns = table_columns
            table_columns = []
            extracted_table_columns = 0
            if not os.path.isdir(os.path.dirname(os.path.abspath(__file__)) + "\\csvs\\" + name + "\\" + season):
                os.makedirs(os.path.dirname(os.path.abspath(__file__)) + "\\csvs\\" + name + "\\" + season)

            df.to_csv(os.path.dirname(os.path.abspath(__file__)) + "\\csvs\\" + name +"\\" + season +"\\"+ button + ".csv", encoding='utf-8')

            if not os.path.isdir(os.path.dirname(os.path.abspath(__file__)) + "\\csvs\\" + name + "\\" + season + "\\htmls"):
                os.makedirs(os.path.dirname(os.path.abspath(__file__)) + "\\csvs\\" + name + "\\" + season + "\\htmls")

            with open(os.path.dirname(os.path.abspath(__file__)) + "\\csvs\\" + name +"\\" + season +"\\htmls\\" + button + ".html", 'w', encoding='utf-8') as f:
                f.write(data_player)

            #      print('Extracted info for player {0}, season {1}'.format(name, season))

            # instead of making silly list with name in data, lets just write the frames to .csv :D
            # maybe filename should be name+season.csv.
            # And maybe we should add the column names before writing to file, for some reason they arent in the table
            #players_and_stats.append({'Name': name, 'Data' + season: df.copy()})
        time.sleep(2) # probably can be shorter i guess

