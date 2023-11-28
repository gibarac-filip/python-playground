import requests
# import json
from bs4 import BeautifulSoup
import pandas as pd
import re

# url = 'https://www.nfl.com/stats/player-stats/category/punts/2023/REG/all/puntingnetyardage/DESC'
# html = requests.get(url).text
# soup = BeautifulSoup(html, 'html5lib')

scoring = {'punt': 0.25,
           'net_yards': 0.1,
           'block': -20.0,
           'touchback': 1.0,
           'in_20': 3.0,
           'oob': 5.0,
           'touchdown': -6.0,
           'sixtyfour': 5.0,
           'sixtynine': 10.0,
           'seventyfour': 15.0,
           'seventyfive': 20.0
           }

punters = { 'Arizona Cardinals': ['Blake Gillikin','Nolan Cooney'],
            'Atlanta Falcons': ['Bradley Pinion'],
            'Baltimore Ravens': ['Jordan Stout'],
            'Buffalo Bills': ['Sam Martin'],
            'Carolina Panthers': ['Johnny Hekker'],
            'Chicago Bears': ['Trenton Gill'],
            'Cincinnati Bengals': ['Brad Robbins'],
            'Cleveland Browns': ['Corey Bojorquez'],
            'Dallas Cowboys': ['Bryan Anger'],
            'Denver Broncos': ['Riley Dixon'],
            'Detroit Lions': ['Jack Fox'],
            'Green Bay Packers': ['Daniel Whelan'],
            'Houston Texans': ['Cameron Johnston', 'Ty Zentner'],
            'Indianapolis Colts': ['Rigoberto Sanchez'],
            'Jacksonville Jaguars': ['Logan Cooke'],
            'Kansas City Chiefs': ['Tommy Townsend'],
            'Las Vegas Raiders': ['AJ Cole'],
            'Los Angeles Chargers': ['J.K. Scott'],
            'Los Angeles Rams': ['Ethan Evans'],
            'Miami Dolphins': ['Jake Bailey'],
            'Minnesota Vikings': ['Ryan Wright'],
            'New England Patriots': ['Bryce Baringer'],
            'New Orleans Saints': ['Lou Hedley'],
            'New York Giants': ['Jamie Gillan'],
            'New York Jets': ['Thomas Morstead'],
            'Philadelphia Eagles': ['Braden Mann', 'Arryn Siposs'],
            'Pittsburgh Steelers': ['Pressley Harvin III', 'Brad Wing'],
            'San Francisco 49ers': ['Mitch Wishnowsky'],
            'Seattle Seahawks': ['Michael Dickson'],
            'Tampa Bay Buccaneers': ['Jake Camarda'],
            'Tennessee Titans': ['Ryan Stonehouse'],
            'Washington Commanders': ['Tress Way']
           }

urls = {}

for team, players in punters.items():
    if team == 'Las Vegas Raiders':
        urls[team] = ['https://www.nfl.com/players/a-j-cole/stats/']
    elif team == 'Los Angeles Chargers':
        urls[team] = ['https://www.nfl.com/players/j-k-scott/stats/']
    # needs better exception handling
    else:
        urls[team] = [f'https://www.nfl.com/players/{player.lower().replace(" ", "-")}/stats/' for player in players]

# Define column headers
headers = ['Week', 'Opponent', 'Result', 'Punts', 'Yards', 'Net Yards', 'Longest Punt',
           'Average Yards per Punt', 'Average Net Yards per Punt', 'Blocked Punt',
           'OOB', 'Downed Punt', 'In the 20', 'Touchback', 'Fair Catches', 'Punts Returned',
           'Punt Return Yards', 'Touchdowns']

# test_teams = ['Atlanta Falcons', 'Washington Commanders']
# test_urls = {team: urls[team] for team in test_teams if team in urls}

# test_teams_2 = ['Houston Texans', 'Pittsburgh Steelers']
# test_urls = {team: urls[team] for team in test_teams_2 if team in urls}

scores = {}

for team, punter in urls.items():
    if len(punter) != 1:
        
        punter_agg = []
        
        for i in punter:
            html = requests.get(i).text
            soup = BeautifulSoup(html, 'html5lib')
            content_divs = soup.find_all('div', class_='d3-o-table--horizontal-scroll')
            
            # Create an empty list to store the data
            data = []

            # Loop through each div
            for div in content_divs:
                tbody_elements = div.find_all('tbody')
                
                # Loop through each tbody
                for tbody in tbody_elements:
                    td_elements = tbody.find_all('td')
                    
                    # Create a list for each row
                    row_data = []
                    
                    # Loop through each td and append its content to the row
                    for td in td_elements:
                        row_data.append(td.get_text())  # Store the content of td
                        
                    data.append(row_data)  # Append the row data to the main data list

            # Convert the list of lists into a Pandas DataFrame
            empty_matrix = pd.DataFrame(data[0])
            punter_agg.append(pd.DataFrame(empty_matrix.values.reshape(-1, 18), columns=headers))
        
        scores[team] = pd.concat(punter_agg, ignore_index=True)
        
    else:
        html = requests.get(punter[0]).text
        soup = BeautifulSoup(html, 'html5lib')
        content_divs = soup.find_all('div', class_='d3-o-table--horizontal-scroll')
        
        # Create an empty list to store the data
        data = []

        # Loop through each div
        for div in content_divs:
            tbody_elements = div.find_all('tbody')
            
            # Loop through each tbody
            for tbody in tbody_elements:
                td_elements = tbody.find_all('td')
                
                # Create a list for each row
                row_data = []
                
                # Loop through each td and append its content to the row
                for td in td_elements:
                    row_data.append(td.get_text())  # Store the content of td
                    
                data.append(row_data)  # Append the row data to the main data list

        # Convert the list of lists into a Pandas DataFrame
        empty_matrix = pd.DataFrame(data[0])
        scores[team] = pd.DataFrame(empty_matrix.values.reshape(-1, 18), columns=headers)

# test_1 = scores['Atlanta Falcons'].iloc[0:3,0:]
# test_2 = scores['Washington Commanders'].iloc[3:6,0:]
# combined_df = pd.concat([test_1, test_2], ignore_index=True)

for _, table in scores.items():
    for i in range(len(table)):
        if type(table['Punts'].iloc[min(i, len(table['Punts']))]) == str:
            table.drop(table[table['Punts'].str.strip() == ''].index, inplace=True)
        else: 
            pass

for key, table in scores.items():
    # Assuming the column 'column_name' contains strings that represent numbers
    # Convert the column values to integers for sorting
    table['Week'] = table['Week'].astype(int)
    table['Punts'] = table['Punts'].astype(float)
    table['Net Yards'] = table['Net Yards'].astype(float)
    table['Blocked Punt'] = table['Blocked Punt'].astype(float)
    table['Touchback'] = table['Touchback'].astype(float)
    table['In the 20'] = table['In the 20'].astype(float)
    table['OOB'] = table['OOB'].astype(float)
    table['Touchdowns'] = table['Touchdowns'].astype(float)
    
    # Sort the table based on the specified column
    scores[key] = table.sort_values(by='Week')

for week, table in scores.items():
    table['Base Score'] = table.apply(lambda row: (
        scoring['punt'] * row['Punts'] +
        scoring['net_yards'] * row['Net Yards'] +
        scoring['block'] * row['Blocked Punt'] +
        scoring['touchback'] * row['Touchback'] +
        scoring['in_20'] * row['In the 20'] +
        scoring['oob'] * row['OOB'] +
        scoring['touchdown'] * row['Touchdowns']
    ), axis=1)

###############################################################
# TESTING
###############################################################


url = 'https://www.nfl.com/games/raiders-at-broncos-2023-reg-1?active-tab=plays'
html = requests.get(url).text
soup = BeautifulSoup(html, 'html5lib')
content_divs = soup.find_all('div', class_='d3-l-wrap')
content_divs[0].find_all('punt')
data = []

result = soup.find('div', class_='d3-l-wrap').find('main').find('div').find('div')

# Loop through each div
for div in content_divs:
    div_1 = div.find_all('main')
    for div in div_1:
        div_2 = div.find_all('div', class_='nfl-c-page')
        if div_2:
            # Get the value of the data-json attribute
            json_data = div_2['data-json']
            data.append(json_data)