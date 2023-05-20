



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
from sklearn.ensemble import RandomForestRegressor

# For time stamps
from datetime import datetime
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.metrics import mean_squared_error, r2_score
from keras.layers import Layer
from keras import backend as K
 

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
clrs = [ 'red', 'blue','green', 'orange']

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

class RBFLayer(Layer):
    def __init__(self, units, gamma, **kwargs):
        super(RBFLayer, self).__init__(**kwargs)
        self.units = units
        self.gamma = K.cast_to_floatx(gamma)

    def build(self, input_shape):
        self.mu = self.add_weight(name='mu',
                                  shape=(int(input_shape[1]), self.units),
                                  initializer='uniform',
                                  trainable=True)
        super(RBFLayer, self).build(input_shape)

    def call(self, inputs):
        diff = K.expand_dims(inputs) - self.mu
        l2 = K.sum(K.pow(diff,2), axis=1)
        res = K.exp(-1 * self.gamma * l2)
        return res

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)



for val in stocks:
    print("\n\n**************"+val+"**************\n\n")

    df=yf.download(val, start='2012-01-01', end=datetime.now())

    df.head()

    df.tail()

    df1=df['Close']

    df1
    plt.plot(df1)
    plt.title(val+' Dataset')
    plt.ylabel('Stock Price')
    plt.xlabel('Time(yr)')
    plt.legend(['Stock Price'], loc='upper left')
    plt.savefig('Graphs_Single_LSTM/'+val+' Dataset'+'.png')
    plt.cla()
    # plt.show()

    ### LSTM are sensitive to the scale of the data. so we apply MinMax scaler
    df1

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
   

    # train_data



    # reshape into X=t,t+1,t+2,t+3 and Y=t+4
    time_step = 100
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, ytest = create_dataset(test_data, time_step)

    print(X_train.shape), print(y_train.shape)

    print(X_test.shape), print(ytest.shape)

    # reshape input to be [samples, time steps, features] which is required for LSTM
    X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)


    # Bi-lstm Part
    batch=32
    epochs=50
    lr=0.01
    optim= "Adam"
    units=100
    model_Bilstm=Sequential()
    model_Bilstm.add(Bidirectional(LSTM(units, activation='sigmoid'), input_shape=(100, 1)))
    model_Bilstm.add(Dense(1))

    opt=tf.keras.optimizers.Adam(
    learning_rate=lr,
    name=optim,)
    model_Bilstm.compile(loss='mean_squared_error',optimizer=opt)
    hist_Bilstm=model_Bilstm.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=epochs,batch_size=batch,verbose=1)
    test_predict_Bilstm=model_Bilstm.predict(X_test)
    test_predict_Bilstm=scaler.inverse_transform(test_predict_Bilstm)
   
   
    ####### LSTM PART
    batch=32
    epochs=30
    lr=0.01
    optim= "Adam" 
    units=100
    model_lstm=Sequential()
    model_lstm.add(LSTM(units,input_shape=(100,1), activation='sigmoid'))
    model_lstm.add(Dense(1))
    opt=tf.keras.optimizers.Adam(
    learning_rate=lr,
    name=optim,)
    model_lstm.compile(loss='mean_squared_error',optimizer=opt)
    hist=model_lstm.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=epochs,batch_size=batch,verbose=1)
    test_predict_lstm=model_lstm.predict(X_test)
    test_predict_lstm=scaler.inverse_transform(test_predict_lstm)


    # RBF model
    batch=32
    epochs=30
    lr=0.01
    optim= "Adam" 
    units=20
    model_rbf = Sequential()
    model_rbf.add(Dense(units, input_shape=(100,)))
    model_rbf.add(RBFLayer(30, 0.5))
    model_rbf.add(RBFLayer(10, 0.2))
    model_rbf.add(Dense(1))
    opt=tf.keras.optimizers.Adam(
    learning_rate=lr,
    name=optim,)
    model_rbf.compile(loss='mean_squared_error',optimizer=opt)
    hist_rbf=model_rbf.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=epochs,batch_size=batch,verbose=0)
    test_predict_rbf=model_rbf.predict(X_test)
    test_predict_rbf=scaler.inverse_transform(test_predict_rbf)

    



    ## RFR model

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1])

    # Train the Random Forest model
    regressor = RandomForestRegressor(
        n_estimators=100,
        random_state=0,
        criterion='squared_error',
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='auto'
        )
    hist=regressor.fit(X_train, y_train.ravel())
    test_predict_rfr=regressor.predict(X_test)
    test_predict_rfr=scaler.inverse_transform(test_predict_rfr.reshape(-1, 1))


    ## Draw Test Graph
    size=len(test_predict_lstm)
    plt.plot(scaler.inverse_transform(df1)[-size:],color='black')
    plt.plot(test_predict_lstm,color=clrs[0],)
    plt.plot(test_predict_Bilstm,color=clrs[1],)
    plt.plot(test_predict_rbf,color=clrs[2],)
    plt.plot(test_predict_rfr,color=clrs[3],)
    plt.title('Model Prediction of Test Data '+val)
    plt.ylabel('Stock Price')
    plt.xlabel("Time(days)")
    plt.legend(['Actual', 'LSTM','Bi-LSTM','RBF','RFR'], loc='upper left')
    plt.savefig('Graphs_all_models/'+val+' test'+'.png',format="png",dpi=1200)


