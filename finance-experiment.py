import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Define the start and end dates
start_date = datetime.now() + timedelta(days=-30)
end_date = datetime.now()

nvida = yf.Ticker("NVDA").history(period='max')
amd = yf.Ticker("AMD").history(period='max')
broadcom = yf.Ticker("AVGO").history(period='max')
intel = yf.Ticker("INTC").history(period='max')

# Initialize an empty DataFrame to store the data
nvidia_by_minute = pd.DataFrame()

# Iterate through 7-day periods
while start_date < end_date:
    # Calculate the end date for the current 7-day period
    next_date = start_date + timedelta(days=7)
    
    # Ensure the next_date doesn't exceed the end_date
    if next_date > end_date:
        next_date = end_date
    
    # Fetch 1m data for the current 7-day period
    data = yf.Ticker("NVDA").history(start=start_date, end=next_date, interval='1m')
    
    # Append the fetched data to the nvidia_by_minute DataFrame
    nvidia_by_minute = nvidia_by_minute._append(data)
    
    # Move to the next 7-day period
    start_date = next_date

# Initialize an empty DataFrame to store the data
amd_by_minute = pd.DataFrame()
start_date = datetime.now() + timedelta(days=-30)
end_date = datetime.now()

# Iterate through 7-day periods
while start_date < end_date:
    # Calculate the end date for the current 7-day period
    next_date = start_date + timedelta(days=7)
    
    # Ensure the next_date doesn't exceed the end_date
    if next_date > end_date:
        next_date = end_date
    
    # Fetch 1m data for the current 7-day period
    data = yf.Ticker("AMD").history(start=start_date, end=next_date, interval='1m')
    
    # Append the fetched data to the amd_by_minute DataFrame
    amd_by_minute = amd_by_minute._append(data)
    
    # Move to the next 7-day period
    start_date = next_date

# Initialize an empty DataFrame to store the data
broadcom_by_minute = pd.DataFrame()
start_date = datetime.now() + timedelta(days=-30)
end_date = datetime.now()

# Iterate through 7-day periods
while start_date < end_date:
    # Calculate the end date for the current 7-day period
    next_date = start_date + timedelta(days=7)
    
    # Ensure the next_date doesn't exceed the end_date
    if next_date > end_date:
        next_date = end_date
    
    # Fetch 1m data for the current 7-day period
    data = yf.Ticker("AVGO").history(start=start_date, end=next_date, interval='1m')
    
    # Append the fetched data to the broadcom_by_minute DataFrame
    broadcom_by_minute = broadcom_by_minute._append(data)
    
    # Move to the next 7-day period
    start_date = next_date

# Initialize an empty DataFrame to store the data
intel_by_minute = pd.DataFrame()
start_date = datetime.now() + timedelta(days=-30)
end_date = datetime.now()

# Iterate through 7-day periods
while start_date < end_date:
    # Calculate the end date for the current 7-day period
    next_date = start_date + timedelta(days=7)
    
    # Ensure the next_date doesn't exceed the end_date
    if next_date > end_date:
        next_date = end_date
    
    # Fetch 1m data for the current 7-day period
    data = yf.Ticker("INTC").history(start=start_date, end=next_date, interval='1m')
    
    # Append the fetched data to the intel_by_minute DataFrame
    intel_by_minute = intel_by_minute._append(data)
    
    # Move to the next 7-day period
    start_date = next_date