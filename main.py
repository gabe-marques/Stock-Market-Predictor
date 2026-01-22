import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import urllib.request, json
import os

# Load configs
configs = json.load(open('config.json', 'r'))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load data from data_source
data_source = 'alphavantage' #alphavantage or kaggle 

if data_source == 'alphavantage':
    api_key = configs["data"]["api_key"]
    ticker = "AAL" # American Airlines stock market prices

    # JSON file with all the stock market data for AAL from the last 20 years
    url_str = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=%s&outputsize=full&apikey=%s"%(ticker,api_key)

    # Save data
    file_to_save = 'stock_market_data-%s.csv'%ticker

    # If data is not already saved, grab data from url and store 
    # date, low, high, volume, close, open to a Pandas DataFrame
    if not os.path.exists(file_to_save):
        with urllib.request.urlopen(url_str) as url:
            data = json.loads(url.read().decode())
            # extract stock market data
            data = data['Time Series (Daily)']
            df = pd.DataFrame(columns=['Date','Low','High','Close','Open'])
            for k,v in data.items():
                date = dt.datetime.strptime(k, '%Y-%m-%d')
                data_row = [date.date(),float(v['3. low']),float(v['2. high']),
                            float(v['4. close']),float(v['1. open'])]
                df.loc[-1,:] = data_row
                df.index = df.index + 1
        print('Data saved to : %s'%file_to_save)        
        df.to_csv(file_to_save)

    # If the data is already there, just load it from csv
    else:
        print('File already exists. Loading data from CSV')
        df = pd.read_csv(file_to_save)
    
elif data_source == 'kaggle':
    data_dir = os.path.join(BASE_DIR, configs["data"]["folder_dir"])
    csv_path = os.path.join(data_dir, 'Stocks', 'hpq.us.txt')
    df = pd.read_csv(csv_path, delimiter=',', usecols=['Date','Open','High','Low','Close'])
    print('Loaded data from the Kaggle repository')

else:
    print('Unrecognized data source')

# Sort df by date and view first few rows
df = df.sort_values('Date')
print(df.head())

# Data Visualization
plt.figure(figsize = (18,9))
plt.plot(range(df.shape[0]),(df['Low']+df['High'])/2.0)
plt.xticks(range(0,df.shape[0],500),df['Date'].loc[::500],rotation=45)
plt.xlabel('Date',fontsize=18)
plt.ylabel('Mid Price',fontsize=18)
plt.show()