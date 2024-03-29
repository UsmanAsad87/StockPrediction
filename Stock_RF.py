

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
# tf.__version__



# stocks=['AAPL']
stocks=['AAPL','MSFT','NVDA','AMZN']
colors = {
    'AAPL': 'red',
    'MSFT': 'blue',
    'NVDA': 'green',
    'AMZN': 'orange'
}

#Defining MAPE function
def MAPE(Y_actual,Y_Predicted):
    mape=mapeE(Y_actual,Y_Predicted)
    return mape

# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)



for val in stocks:
    print("\n\n**************"+val+"**************\n\n")

    df=yf.download(val, start='2012-01-01', end=datetime.now())

    df.head()

    df.tail()

    df1=df['Close'] 

    plt.plot(df1)
    plt.title(val+' Dataset')
    plt.ylabel('Stock Price')
    plt.xlabel('Time(yr)')
    plt.legend(['Stock Price'], loc='upper left')
    plt.savefig('Graphs_RF/'+val+' Dataset'+'.png')
    plt.cla()
    # plt.show()



    from sklearn.preprocessing import MinMaxScaler
    scaler=MinMaxScaler(feature_range=(0,1))
    df1=scaler.fit_transform(np.array(df1).reshape(-1,1))

    ##splitting dataset into train and test split
    training_size=int(len(df1)*0.80)
    test_size=len(df1)-training_size
    # train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]
    train_data,test_data,last30=df1[0:training_size,:],df1[training_size:len(df1)-30,:1],df1[len(df1)-30:len(df1),:]


    print("Train Data Size: "+str(training_size))
    print("Test Data Size: "+str(test_size))
   

    # train_data



    # reshape into X=t,t+1,t+2,t+3 and Y=t+4
    time_step = 100
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, ytest = create_dataset(test_data, time_step)

    print(X_train.shape), print(y_train.shape)

    print(X_test.shape), print(ytest.shape)

    # reshape input to be [samples, time steps, features] which is required for LSTM
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1])


    # Train the Random Forest model
    regressor = RandomForestRegressor(
        n_estimators=100,
        random_state=0,
        # criterion='squared_error',
        # max_depth=None,
        # min_samples_split=2,
        # min_samples_leaf=1,
        # max_features='auto'
        )
    hist=regressor.fit(X_train, y_train.ravel())

    ### Lets Do the prediction and check performance metrics
    train_predict=regressor.predict(X_train)
    test_predict=regressor.predict(X_test)

    #Transformback to original form
    train_predict=scaler.inverse_transform(train_predict.reshape(-1, 1))
    test_predict=scaler.inverse_transform(test_predict.reshape(-1, 1))

    ### Calculate RMSE performance metrics
    import math

    y_train2=scaler.inverse_transform(y_train.reshape(-1,1))
    y_test2=scaler.inverse_transform(ytest.reshape(-1,1))

    train_err=math.sqrt(mean_squared_error(y_train2,train_predict))
    print("Train RMSE: "+str(train_err))
    print("Train MSE: "+str(mean_squared_error(y_train2,train_predict)))
    print("Train MAPE: "+str(MAPE(y_train2,train_predict)))
    print("Train MAE: "+str(mae(y_train2,train_predict)))
    print("Train R2: " + str(r2_score(y_train2, train_predict)))



    ### Test Data RMSE,MSE,MAPE,MAE
    test_err=math.sqrt(mean_squared_error(y_test2,test_predict))
    print("Test RMSE: "+str(test_err))
    print("Test MSE: "+str(mean_squared_error(y_test2,test_predict)))
    print("Test MAPE: "+str(MAPE(y_test2,test_predict)))
    print("Test MAE: "+str(mae(y_test2,test_predict)))
    print("Test R2: " + str(r2_score(y_test2, test_predict)))

    
   
    ### Plotting 
    # shift train predictions for plotting
    look_back=100
    trainPredictPlot = np.empty_like(df1)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
    # shift test predictions for plotting
    testPredictPlot = np.empty_like(df1)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1 -30, :] = test_predict
    # plot baseline and predictions
    plt.plot(scaler.inverse_transform(df1))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.title('Model Prediction of Test and Train Data of '+val)
    plt.ylabel('Stock Price')
    plt.xlabel("Time(yr)")
    # plt.xlabel('Time(yr)\nTrain RMSE: '+str(train_err)+" \nTest RMSE: "+str(test_err))
    plt.legend(['Stock Price','Train', 'Test'], loc='upper left')
    plt.savefig('Graphs_RF/'+val+' train_test'+'.png')
    plt.cla()
    # plt.show()


    size=len(test_predict)
    plt.plot(scaler.inverse_transform(df1)[-size:])
    plt.plot(test_predict,color='green')
    plt.title('Model Prediction of Test Data '+val)
    plt.ylabel('Stock Price')
    plt.xlabel("Time")
    plt.legend(['Stock Price', 'Test'], loc='upper left')
    plt.savefig('Graphs_RF/'+val+' test'+'.png')
    plt.cla()



    #NEXT 30 days Prediction.
    x_input=test_data[len(test_data)-100:].reshape(1,-1)
    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    temp_input
    lst_output=[]
    n_steps=100
    i=0
    while(i<30):
        if(len(temp_input)>100):
            x_input=np.array(temp_input[1:])
            x_input=x_input.reshape(1,-1)
            # x_input = x_input.reshape((1, n_steps, 1))
            yhat = regressor.predict(x_input)
            temp_input.extend(yhat.tolist())
            temp_input=temp_input[1:]
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            # x_input = x_input.reshape((1, n_steps,1))
            yhat = regressor.predict(x_input)
            temp_input.extend(yhat.tolist())
            lst_output.extend(yhat.tolist())
            i=i+1

    lst_output = np.array(lst_output)
    next_pred=scaler.inverse_transform(lst_output.reshape(1,-1))
    next_pred=str(next_pred[0]).replace('[','').replace(']','').replace("  ",' ')
    next_pred_list=next_pred.split(" ")
    print('Predicted Values: ')
    for y in range(len(next_pred_list)):
        if( y==0 or y==2 or y==6 or y==29):
            print("Day "+str(y+1)+": "+ str(next_pred_list[y]))

    last30=scaler.inverse_transform(last30.reshape(-1,1))
    print('Real Values: ')
    for y in range(len(last30)):
        if( y==0 or y==2 or y==6 or y==29):
            print("Day "+str(y+1)+": "+ str(last30[y]))


