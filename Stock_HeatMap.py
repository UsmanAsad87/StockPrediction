

### Stock Market Prediction And Forecasting Using Stacked LSTM



### Data Collection
import pandas_datareader as pdr
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mapeE

# For reading stock data from yahoo
from pandas_datareader.data import DataReader
import yfinance as yf
import tensorflow as tf
import numpy as np

# For time stamps
from datetime import datetime
import matplotlib.pyplot as plt
from tensorflow import keras
 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from keras.layers import Dropout
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt
# tf.__version__



# stocks=['AAPL']
stocks=['AAPL','MSFT','NVDA','AMZN']





for val in stocks:
    print("\n\n**************"+val+"**************\n\n")
    df=yf.download(val, start='2012-01-01', end=datetime.now())
    df.head()


    # correlation_matrix = df.corr()

    # Create a heatmap
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')

    # # Set the title
    # plt.title('Stock Correlation Heatmap ' + val)
    # plt.savefig('Graphs_LSTM/'+val+' heatMap'+'.png',format="png",dpi=1500)
    # plt.cla()
    # # Show the plot
    # # plt.show()


    # features = [col for col in df.columns if col != 'Adj Close']

    # # Plot all features
    # fig, axes = plt.subplots(len(features), 1, figsize=(10, 6 * len(features)))
    # fig.suptitle(val + " Stock Features", fontsize=16)

    # for i, feature in enumerate(features):
    #     axes[i].plot(df.index, df[feature])
    #     axes[i].set_ylabel(feature)
    #     axes[i].grid(True)

    # plt.xlabel("Date")
    # plt.show()



    # plt.figure(figsize=(10, 6))
    # plt.title(val + " Stock Features")
    # plt.xlabel("Date")
    # plt.ylabel("Value")
    # plt.grid(True)

    # for column in df.columns:
    #     plt.plot(df.index, df[column], label=column)

    # plt.legend()
    # plt.show()



    features = [col for col in df.columns if col != 'Volume']

    # Plot all features except 'Volume'
    plt.figure(figsize=(10, 6))
    plt.title(val + " Stock Features")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.grid(True)
    

    for feature in features:
        plt.plot(df.index, df[feature], label=feature)

    plt.legend()
    plt.savefig('Graphs_Dataset/'+val+' Dataset'+'.png',format="png",dpi=1500)
    # plt.show()

