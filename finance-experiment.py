import yfinance as yf

# Request historical data for past 5 years
nvidia = yf.Ticker("NVDA").history(period='5y')
apple = yf.Ticker("AAPL").history(period='5y')
microsoft = yf.Ticker("MSFT").history(period='5y')

# Show info
print(nvidia.info())
print(apple.info())
print(microsoft.info())