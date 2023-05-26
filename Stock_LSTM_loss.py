

### Stock Market Prediction And Forecasting Using Stacked LSTM



### Data Collection
import pandas_datareader as pdr
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mapeE
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adadelta
# For reading stock data from yahoo
from pandas_datareader.data import DataReader
import yfinance as yf
import tensorflow as tf
import numpy as np

# For time stamps
from datetime import datetime
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.metrics import mean_squared_error, r2_score
 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from keras.layers import Dropout
import tensorflow as tf
# tf.__version__



stocks=['AAPL']
# stocks=['AAPL','MSFT','NVDA','AMZN']
colors = {
    'AAPL': 'red',
    'MSFT': 'blue',
    'NVDA': 'green',
    'AMZN': 'orange'
}

#Defining MAPE function
def MAPE(Y_actual,Y_Predicted):
    # print(Y_actual)
    # Y_actual=Y_actual+0.001
    # Y_Predicted=Y_Predicted+0.001
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



for val in stocks:
    print("\n\n**************"+val+"**************\n\n")

    df=yf.download(val, start='2012-01-01', end=datetime.now())

    df.head()

    df.tail()

    df1=df['Close']



    ### LSTM are sensitive to the scale of the data. so we apply MinMax scaler
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

    print(X_train.shape), print(y_train.shape)

    print(X_test.shape), print(ytest.shape)

    # reshape input to be [samples, time steps, features] which is required for LSTM
    X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)


    
    batch=32
    epochs=30
    lr=0.001
    # optim= "Adam"
    optim= "RMSprop"
    optim= "AdaDelta"
    units=100

    model=Sequential()
    model.add(LSTM(units,input_shape=(100,1), activation='sigmoid'))
    model.add(Dense(1))

    # opt=tf.keras.optimizers.Adam(
    # learning_rate=lr,
    # name=optim,)
   

    # opt = RMSprop(learning_rate=lr, name=optim)
    opt = Adadelta(learning_rate=lr)

    model.compile(loss='mean_squared_error',optimizer=opt)



    



    hist=model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=epochs,batch_size=batch,verbose=0)
    print(hist.history)
    model.save("saved_lstm_model_single/"+val+"/my_model.h5")
    # model = keras.models.load_model("saved_lstm_model_single/"+val+"/my_model.h5")




    ### Lets Do the prediction and check performance metrics
    train_predict=model.predict(X_train)
    test_predict=model.predict(X_test)

    #Transformback to original form
    train_predict=scaler.inverse_transform(train_predict)
    test_predict=scaler.inverse_transform(test_predict)

    ### Calculate RMSE performance metrics
    import math
    from sklearn.metrics import mean_squared_error

    y_train2=scaler.inverse_transform(y_train.reshape(-1,1))
    y_test2=scaler.inverse_transform(ytest.reshape(-1,1))


    
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('Graphs_LSTM_loss/'+val+" -"+str(lr)+"-"+str(optim)+' loss'+'.png')
    # plt.show()




