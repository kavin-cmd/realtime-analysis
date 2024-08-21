import requests
import pandas as pd
import time

def fetch_stock_data(api_key, symbol, interval='1min', retries=3):
    """
    Fetch real-time stock data from Alpha Vantage.
    
    Parameters:
    - api_key: Your Alpha Vantage API key.
    - symbol: The stock symbol (e.g., 'AAPL').
    - interval: Time interval between data points (default '1min').
    - retries: Number of retry attempts for failed API requests (default 3).
    
    Returns:
    - df: Pandas DataFrame containing the stock data.
    """
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval={interval}&apikey={api_key}'
    
    for attempt in range(retries):
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if 'Time Series (1min)' in data:
                df = pd.DataFrame.from_dict(data['Time Series (1min)'], orient='index')
                df.columns = ['open', 'high', 'low', 'close', 'volume']
                df.index = pd.to_datetime(df.index)
                df = df.astype(float)
                return df
            elif 'Note' in data:
                print("API call limit reached. Waiting for 60 seconds before retrying...")
                time.sleep(60)
            else:
                print("Unexpected response format:", data)
        else:
            print(f"Failed to fetch data. Status code: {response.status_code}")
    
    raise Exception("Failed to fetch data after multiple attempts.")
