import requests
from bs4 import BeautifulSoup
import pandas as pd
import requests
from collections import defaultdict

user = defaultdict()

XFL = '916334389204840448'

users = requests.get('https://api.sleeper.app/v1/league/' + XFL + '/users')
users = users.json()

roster = requests.get('http://api.sleeper.app/v1/league/' + XFL + '/rosters')
roster = roster.json()

for user_data in users:
    for roster_data in roster:
        if roster_data['owner_id']  == user_data['user_id']:
            user[sorted(roster_data['players'])[-1]] = user_data['metadata']['team_name']
        else:
            continue
    continue

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

# TODO: create code that picks up punters and assigns them to the team
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

# TODO: Needs a little more work here to figure out blanks 
for _, table in scores.items():
    for i in range(len(table)):
        try:
            if type(table['Punts'].iloc[min(i, len(table['Punts']))]) == str:
                table.drop(table[table['Punts'].str.strip() == ''].index, inplace=True)
            else: 
                pass
        except: table['Punts'].iloc[min(i, len(table['Punts']))-1] = 0

for key, table in scores.items():
    # Assuming the column 'column_name' contains strings that represent numbers
    # Convert the column values to integers for sorting
    try: table['Week'] = table['Week'].astype(int)
    except: table['Week'] = 0
    
    try: table['Punts'] = table['Punts'].astype(int)
    except: table['Punts'] = 0
    
    try: table['Longest Punt'] = table['Longest Punt'].astype(float)
    except: table['Longest Punt'] = 0

    try: table['Net Yards'] = table['Net Yards'].astype(float)
    except: table['Net Yards'] = 0
    
    try: table['Blocked Punt'] = table['Blocked Punt'].astype(float)
    except: table['Blocked Punt'] = 0
    
    try: table['Touchback'] = table['Touchback'].astype(float)
    except: table['Touchback'] = 0
    
    try: table['In the 20'] = table['In the 20'].astype(float)
    except: table['In the 20'] = 0
    
    try: table['OOB'] = table['OOB'].astype(float)
    except: table['OOB'] = 0
    
    try: table['Touchdowns'] = table['Touchdowns'].astype(float)
    except: table['Touchdowns'] = 0
    
    # Sort the table based on the specified column
    scores[key] = table.sort_values(by='Week')

for week, table in scores.items():
    def calculate_base_score(row):
        base_score = (
            scoring['punt'] * row['Punts'] +
            scoring['net_yards'] * row['Net Yards'] +
            scoring['block'] * row['Blocked Punt'] +
            scoring['touchback'] * row['Touchback'] +
            scoring['in_20'] * row['In the 20'] +
            scoring['oob'] * row['OOB'] +
            scoring['touchdown'] * row['Touchdowns']
        )
        if 60 <= row['Longest Punt'] <= 64:
            base_score += scoring['sixtyfour']
        elif 65 <= row['Longest Punt'] <= 69:
            base_score += scoring['sixtynine']
        elif 70 <= row['Longest Punt'] <= 74:
            base_score += scoring['seventyfour']
        elif 75 <= row['Longest Punt']:
            base_score += scoring['seventyfive']
        return base_score

    table['Base Score'] = table.apply(calculate_base_score, axis=1)

data = {
    'Team': [],
    'Week': [],
    'Longest Punt': [],
    'Base Score': [],
    'Manual Check': []
}

for team, score_df in scores.items():
    data['Team'].extend([team] * len(score_df))
    data['Week'].extend(score_df['Week'])
    data['Longest Punt'].extend(score_df['Longest Punt'])
    data['Base Score'].extend(score_df['Base Score'])
    data['Manual Check'].extend(['Needs Manually Updated Score' if punt > 59 else '' for punt in score_df['Longest Punt']])

df = pd.DataFrame(data)
df[df['Week']==max(df['Week'])].to_excel('Week-'+ str(max(df['Week'])) +'.xlsx', index=False)

###############################################################
# TESTING
###############################################################
"""
# nfl is a dud, doesn't let me access their shit
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
            
url = 'https://www.pro-football-reference.com/boxscores/202309100den.htm'
html = requests.get(url).text
soup = BeautifulSoup(html, 'html5lib')
content_divs = soup.find('body').find('div', class_='box').find('div', class_='table_wrapper').find('tbody')

specific_text = soup.find(lambda tag: tag.name == 'div' and tag.text.strip() == 'punt')

if specific_text:
    print(specific_text.get_text())
"""