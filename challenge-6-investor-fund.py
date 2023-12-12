import pandas as pd

url = 'https://raw.githubusercontent.com/datasets/investor-flow-of-funds-us/master/data/weekly.csv'
df = pd.read_csv(url)

# 1. What is the frequency of the dataset? (The time period between each row)

# Convert the column to datetime if it's not already in datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Sort the DataFrame by the date column (if it's not already sorted)
df = df.sort_values('Date')

# Calculate the time differences between consecutive rows
df['frequency'] = df['Date'].diff()

# Display the DataFrame with the frequency column
print(df)
    
# 2. What is the data type of the index?
type(df.index)
# 3. Set the index to a Datetime.

# Set the 'Date' column as the index
df.set_index('Date', inplace=True)

# Display the DataFrame with 'Date' column as the index
print(df)

# 4. Change the frequency to monthly, sum the values and assign it to new variable called monthly.

# Resample to monthly frequency and sum the values
monthly = df[:].resample('M').sum()

# Display the 'monthly' variable containing the summed values on a monthly frequency
print(monthly)

# 5. You will notice that it filled the dataFrame with months that don’t have any data with NaN. Let’s drop these rows.
monthly = monthly.dropna()

# 6. Good, now we have the monthly data. Now change the frequency to year and assign to a new variable called year.

# Resample to yearly frequency and sum the values
yearly = df[:].resample('Y').sum()
yearly = yearly.dropna()

# Display the 'yearly' variable containing the summed values on a yearly frequency
print(yearly)

# 7. Create your own question and answer it.