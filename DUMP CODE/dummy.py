

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
from sklearn.model_selection import GridSearchCV

import tensorflow as tf
# tf.__version__


stocks=['AAPL']
#stocks=['AAPL','MSFT','GOOG','AMZN']

#Defining MAPE function
def MAPE(Y_actual,Y_Predicted):
    # print(Y_actual)
    mape=mapeE(Y_actual,Y_Predicted)
    # mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))*100
    return mape

# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)



def create_model(lr=0.001, optimizer='Adam', batch_size=16, epochs=20, units=100):
    model=Sequential()
    model.add(LSTM(units, input_shape=(100,1), activation='sigmoid'))
    model.add(Dense(1))
    opt = None
    if optimizer == 'Adam':
        opt=tf.keras.optimizers.Adam(learning_rate=lr, name="Adam")
    elif optimizer == 'SGD':
        opt=tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9, name="SGD")
    elif optimizer == 'RMSprop':
        opt=tf.keras.optimizers.RMSprop(learning_rate=lr, momentum=0.9, name="RMSprop")
    model.compile(loss='mean_squared_error', optimizer=opt)
    history=model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=epochs,batch_size=batch_size,verbose=0)
    return model

for val in stocks:
    print("\n\n**************"+val+"**************\n\n")

    df=yf.download(val, start='2012-01-01', end=datetime.now())

    df.head()

    df.tail()

    df1=df['Close']

   


    from sklearn.preprocessing import MinMaxScaler
    scaler=MinMaxScaler(feature_range=(0,1))
    df1=scaler.fit_transform(np.array(df1).reshape(-1,1))

    # print(df1)

    ##splitting dataset into train and test split
    training_size=int(len(df1)*0.80)
    test_size=len(df1)-training_size
    train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]

    print("Train Data Size: "+str(training_size))
    print("Test Data Size: "+str(test_size))
   



    # reshape into X=t,t+1,t+2,t+3 and Y=t+4
    time_step = 100
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, ytest = create_dataset(test_data, time_step)


    # reshape input to be [samples, time steps, features] which is required for LSTM
    X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

    # ### Create the Stacked LSTM model

    
    # model.summary()

   
    param_grid = {
    'lr': [0.001, 0.01, 0.1],
    'optimizer': ['Adam', 'SGD', 'RMSprop'],
    'batch_size': [16, 32, 64],
    'epochs': [10, 20, 30],
    'units': [50, 100, 200]
    }
    model = keras.wrappers.scikit_learn.KerasRegressor(build_fn=create_model, verbose=0)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3, n_jobs=-1)
    grid_result = grid_search.fit(X_train, y_train)



    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


    



    # ### Lets Do the prediction and check performance metrics
    # train_predict=model.predict(X_train)
    # test_predict=model.predict(X_test)


    # ### Calculate RMSE performance metrics
    # import math
    # from sklearn.metrics import mean_squared_error

    # y_train2=y_train.reshape(-1,1)
    # y_test2=ytest.reshape(-1,1)

    # ### Train Data RMSE,MSE,MAPE,MAE
    # train_err=math.sqrt(mean_squared_error(y_train2,train_predict))
    # print("Train RMSE: "+str(train_err))
    # print("Train MSE: "+str(mean_squared_error(y_train2,train_predict)))
    # print("Train MAPE: "+str(MAPE(y_train2,train_predict)))
    # print("Train MAE: "+str(mae(y_train2,train_predict)))



    # ### Test Data RMSE,MSE,MAPE,MAE
    # test_err=math.sqrt(mean_squared_error(y_test2,test_predict))
    # print("Test RMSE: "+str(test_err))
    # print("Test MSE: "+str(mean_squared_error(y_test2,test_predict)))
    # print("Test MAPE: "+str(MAPE(y_test2,test_predict)))
    # print("Test MAE: "+str(mae(y_test2,test_predict)))
