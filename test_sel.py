import time
from datetime import datetime
import requests, re
from selenium import webdriver
from selenium.webdriver.common.by import By
import pandas as pd
import os
from bs4 import BeautifulSoup, Comment

seasons = ["2022-2023", '2021-2022', '2020-2021', '2019-2020']
player_cards = []

buttons = ["Summary", "Passing", "Pass Types", "Goal and Shot Creation", "Defensive Actions", "Possession",
           "Miscellaneous Stats"]  # by default its summary
table_columns_summary = []
table_columns_passing = []
table_columns_passtypes = []
table_columns_goalandshotcreation = []
table_columns_defensiveactions = []
table_columns_possesion = []
table_columns_miscallaneous = []
got_tables_columns = False
tables_got_count = 0

driver = webdriver.Chrome()
button_text_close = "DISAGREE"
disagree_already_pressed = False

continue_with_players = False

for season in seasons:
    URL_season = 'https://fbref.com/en/comps/9/' + season + '/stats/' + season + '-Premier-League-Stats'
    page = requests.get(URL_season).content
    soup = BeautifulSoup(page, "html.parser")
    comments = soup.find_all(string=lambda text: isinstance(text, Comment))
    links = []
    for each in comments:
        if 'table' in each:
            links.append(re.findall(r'(en/players/[^\s]+)', str(each)))
    # (There are two for each player, one to the player profile, and one that shows match log for given season)
    # We only really need the match logs i guess, maybe the player-cards could be useful later.
    match_logs = links[0][1::2]
    player_cards = links[0][0::2]
    cleaned_match_logs = [link.split('"')[0] for link in match_logs]  # Getting just the link

    time.sleep(2)

    for i, player in enumerate(player_cards):  # when getting all players remove ':5'
        print('season:{0}, progress in season:{1}/{2}'.format(season, i + 1, len(player_cards)))
        start_time = datetime.now()
        name = player.split("\"")[0].split('/')[-1]
        URL_player = 'https://fbref.com/' + cleaned_match_logs[i]

        if ('Darko-Gyabi' in name):
            continue_with_players =True
        if not continue_with_players: continue

        driver.get(URL_player)
        if not disagree_already_pressed:
            disagree_button = driver.find_element("xpath", "//button/span[text()='%s']" % button_text_close)
            # Close the pop-up
            disagree_button.click()
            disagree_already_pressed = True
        df = pd.DataFrame()
        for button in buttons:
            if button != buttons[0]:
                print(button)
                element = driver.find_element(By.XPATH, f"//a[text()='{button}']")
                driver.execute_script("arguments[0].scrollIntoView();", element)
                driver.execute_script("arguments[0].click();", element)

            data_player = driver.page_source

            table = driver.find_element(By.ID, "div_matchlogs_all")
            table_data = []

            if not got_tables_columns:
                tablecolumns = []
                table_heads = table.find_elements(By.TAG_NAME, "thead")
                for table_head in table_heads:
                    table_trs = table_head.find_elements(By.TAG_NAME, "tr")
                    for table_tr in table_trs:
                        if "over_header" in table_tr.get_attribute("class"): continue
                        table_ths = table_tr.find_elements(By.TAG_NAME, "th")
                        for table_th in table_ths:
                            tablecolumns.append(table_th.text)
                if (button == buttons[0]):
                    table_columns_summary = tablecolumns
                elif (button == buttons[1]):
                    table_columns_passing = tablecolumns
                elif (button == buttons[2]):
                    table_columns_passtypes = tablecolumns
                elif (button == buttons[3]):
                    table_columns_goalandshotcreation = tablecolumns
                elif (button == buttons[4]):
                    table_columns_defensiveactions = tablecolumns
                elif (button == buttons[5]):
                    table_columns_possesion = tablecolumns
                elif (button == buttons[6]):
                    table_columns_miscallaneous = tablecolumns
                tables_got_count += 1
                if tables_got_count == 7:
                    got_tables_columns = True
            dates = []
            table_data = []

            rows = table.find_elements(By.XPATH, '//tbody/tr[not(contains(@class, "unused"))]')

            for row in rows:
                # Extract data from <th> elements (dates)
                dates_th = row.find_elements(By.TAG_NAME, "th")
                dates = [date.text for date in dates_th]

                # Extract data from <td> elements
                cells = row.find_elements(By.TAG_NAME, "td")
                row_data = []

                for cell in cells:
                    if cell.text == 'Match Report':
                        # Extract the href link from the "Match Report" cell
                        href_element = cell.find_element(By.TAG_NAME, 'a')
                        row_data.append(href_element.get_attribute('href'))
                    else:
                        row_data.append(cell.text)

                # Combine the dates and row_data
                row_data = dates + row_data
                table_data.append(row_data)

            # table_tags = table.find_elements(By.TAG_NAME, "tbody")
            # for table_tag in table_tags:
            #     rows = table_tag.find_elements(By.TAG_NAME, "tr")
            #     for index, row in enumerate(rows):
            #         if "unused" in row.get_attribute("class"): continue
            #         row_data = []
            #         # Iterate through row cells (td elements)
            #         dates_th = row.find_elements(By.TAG_NAME, "th")
            #         for date in dates_th:
            #             dates.append(date.text)
            #         cells = row.find_elements(By.TAG_NAME, "td")
            #         for cell in cells:
            #             if cell.text == 'Match Report':
            #                 href_element = cell.find_element(By.TAG_NAME, 'a')
            #                 row_data.append(href_element.get_attribute('href'))
            #             else:
            #                 row_data.append(cell.text)
            #         table_data.append(row_data)

            # for index, date in enumerate(dates):
            #     table_data[index].insert(0, dates[index])

            inter_dataframe = pd.DataFrame(table_data)

            if (button == buttons[0]):
                tablecolumns =  table_columns_summary
            elif (button == buttons[1]):
                tablecolumns = table_columns_passing
            elif (button == buttons[2]):
                tablecolumns = table_columns_passtypes
            elif (button == buttons[3]):
                tablecolumns = table_columns_goalandshotcreation
            elif (button == buttons[4]):
                tablecolumns = table_columns_defensiveactions
            elif (button == buttons[5]):
                tablecolumns = table_columns_possesion
            elif (button == buttons[6]):
                tablecolumns = table_columns_miscallaneous


            inter_dataframe.columns = tablecolumns
            df = pd.concat([df, inter_dataframe], axis=1)
        df = df.loc[:, ~df.columns.duplicated()]
        if not os.path.isdir(os.path.dirname(os.path.abspath(__file__)) + "\\csvs\\" + name):
            os.makedirs(os.path.dirname(os.path.abspath(__file__)) + "\\csvs\\" + name)
        df.to_csv(os.path.dirname(os.path.abspath(__file__)) + "\\csvs\\" + name + "\\" + season + ".csv",
                  encoding='utf-8')
        end_time = datetime.now()
        print("Player time: ")
        print((end_time - start_time).seconds)
driver.quit()