import pandas as pd
import math as math

url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/drinks.csv'
drinks = pd.read_csv(url, sep =',')

# 1. Which continent drinks more beer on average?
drinks.groupby('continent')['beer_servings'].mean().reset_index().rename(columns={'beer_servings': 'avg_beer_servings'})

# 2. For each continent print the statistics for wine consumption.

# Define a function to calculate the 25th and 75th percentiles
def percentile_25(x):
    return x.quantile(0.25)

def percentile_75(x):
    return x.quantile(0.75)

# Group by continent and calculate mean, median, mode, 25th, and 75th percentiles in a single operation
cont_wine = drinks.groupby('continent')['wine_servings'].agg(['mean', 'median', lambda x: x.mode().iloc[0], percentile_25, percentile_75]).reset_index()

# Rename columns
cont_wine.columns = ['continent', 'avg_wine_servings', 'median_wine_servings', 'mode_wine_servings', '25th_percentile', '75th_percentile']

# Round to 2 decimal places  
cont_wine['avg_wine_servings'] = round(cont_wine['avg_wine_servings'], 2)

cont_wine

# 3. Print the mean and median alcohol consumption per continent for every column.
# Group by continent and calculate mean, median, mode, 25th, and 75th percentiles in a single operation
cont_alc = drinks.groupby('continent')['total_litres_of_pure_alcohol'].agg(['mean', 'median', lambda x: x.mode().iloc[0], percentile_25, percentile_75]).reset_index()

# Rename columns
cont_alc.columns = ['continent', 'avg_alc_servings', 'median_alc_servings', 'mode_alc_servings', '25th_percentile', '75th_percentile']

# Round to 2 decimal places  
cont_alc['avg_alc_servings'] = round(cont_alc['avg_alc_servings'], 2)

cont_alc
