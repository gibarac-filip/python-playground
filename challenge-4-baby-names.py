import pandas as pd
import math as math

url = 'challenge-4.csv'
baby_names = pd.read_csv(url, sep =',')

# 1. See the first 10 entries.
baby_names.iloc[0:10, :]

# 2. Delete the columns ‘Unnamed: 0’ and ‘Id’.
names = baby_names[["Name", "Year", "Gender", "State", "Count"]]

# 3. Group the dataset by name, assign to a variable called names, and sort the dataset by highest to lowest count.
names = baby_names.groupby('Name')['Count'].sum().reset_index()
names = names.sort_values(by='Count', ascending=False)

# 4. How many different names exist in the dataset?
names.nunique(dropna=True)[0]

# 5. What is the name with most occurrences?
# names[names["Count"] == max(names["Count"])]["Name"]
names.loc[names['Count'].idxmax(), 'Name']

# 6. What is the standard deviation of count of names?
round(names["Count"].std(),2)

# 7. Get a summary of the dataset with the mean, min, max, std and quartiles.
names["Count"].describe()
