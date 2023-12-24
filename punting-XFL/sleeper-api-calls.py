import requests
import json

XFL = '916334389204840448'
LEOG = '991751911072206848'

draft = requests.get('http://api.sleeper.app/v1/league/' + XFL + '/drafts')
draft = draft.json()

user = requests.get('http://api.sleeper.app/v1/user/FillyCheeseSteak')
user = user.json()

league = requests.get('https://api.sleeper.app/v1/user/' + user['user_id'] + '/leagues/nfl/2023')
league = league.json()

roster = requests.get('http://api.sleeper.app/v1/league/' + XFL + '/rosters')
roster = roster.json()