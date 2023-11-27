import pandas as pd
import math as math

url = 'https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/02_Filtering_%26_Sorting/Euro12/Euro_2012_stats_TEAM.csv'
evro = pd.read_csv(url, sep =',')

# 1. How many teams participated in the Euro2012?
evro["Team"].nunique(dropna=True)

# 2. What is the number of columns in the dataset?
len(evro.iloc[0,0:])

# 3. View only the columns Team, Yellow Cards and Red Cards and assign them to a dataframe called discipline.
discipline = evro.loc[:,["Team", "Yellow Cards", "Red Cards"]]

# 4. Sort the teams by Red Cards, then to Yellow Cards.
discipline.sort_values(by=[ "Red Cards", "Yellow Cards"], ascending=[False, False])

# 5. Calculate the mean Yellow Cards given per Team.
sum(discipline["Yellow Cards"]) / discipline["Team"].nunique(dropna=True)
discipline["Yellow Cards"].mean()

# 6. Filter teams that scored more than 6 goals.
evro[evro["Goals"]>6]

# 7. Select the teams that start with the letter G.
evro[evro["Team"].str.startswith('G')]

# 8. Select the first 7 columns.
evro.iloc[0:,0:7]

# 9. Select all columns except the last 3.
evro.iloc[0:,:-3]

# 10. Present only the Shooting Accuracy from England, Italy and Russia.
evro.loc[evro['Team'].isin(['England', 'Russia', 'Spain']), ['Team', 'Shooting Accuracy']]