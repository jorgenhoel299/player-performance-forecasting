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
seasons = ['2023-2024'] # Add more sesons to list when we are ready
for season in seasons:
    URL_season = 'https://fbref.com/en/comps/9/' + season + '/stats/' + season +'-Premier-League-Stats'
    page = requests.get(URL_season).content
    soup = BeautifulSoup(page, "html.parser")

    # Until you scroll down on the page the tables are commented in the html, so needs to be extracted like this
    comments = soup.find_all(string=lambda text: isinstance(text, Comment))
    tables = []
    for each in comments:
        if 'table' in each:
            try:
                tables.append(pd.read_html(each)[0])
            except:
                continue
    data = tables[0]

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
    for i, player in enumerate(player_cards[:5]): # when getting all players remove ':5'
        print('season:{0}, progress in season:{1}/{2}'.format(season, i+1, len(player_cards)))
        name = player.rsplit('/', 1)[1].split('"')[0] # Getting name of player as str
        URL_player = 'https://fbref.com/' + cleaned_match_logs[i]

        data_player = requests.get(URL_player).text
        soup_player = BeautifulSoup(data_player, 'html.parser')
        # On these urls, the table is simpler to extract
        table_player = soup_player.find('table', class_='min_width sortable stats_table min_width shade_zero')
        # ELEGANTLY making table into array and then into a df
        data = []
        for row in table_player.find_all('tr'):
            row_data = []
            for cell in row.find_all('td'):
                row_data.append(cell.text)
            data.append(row_data)
        df = pd.DataFrame(data)
        print(df.head(3))

        # instead of making silly list with name in data, lets just write the frames to .csv :D
        # maybe filename should be name+season.csv.
        # And maybe we should add the column names before writing to file, for some reason they arent in the table
        players_and_stats.append({'Name': name, 'Data' + season: df.copy()})
        time.sleep(5) # probably can be shorter i guess

#print(players_and_stats)
