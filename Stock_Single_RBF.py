

### Stock Market Prediction And Forecasting Using Stacked LSTM



### Data Collection
import pandas_datareader as pdr
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mapeE



from keras.layers import Layer
from keras import backend as K

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
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
import tensorflow as tf
from keras.layers import Reshape
# tf.__version__



stocks=['AAPL']
#stocks=['AAPL','MSFT','GOOG','AMZN']

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

    ### Create the Stacked LSTM model

    

    
    batch=32
    epochs=10
    lr=0.01
    optim= "Adam" #"SGD" # "RMSprop" #
    units=50

    # # model=Sequential()
    # model = Sequential()
    # model.add(Dense(100, input_shape=(100,)))
    # model.add(RBFLayer(100, 0.5))
    # # model.add(Dropout(.2))
    # model.add(RBFLayer(50, 0.5))
    # # model.add(Dropout(.2))
    # # model.add(RBFLayer(25, 0.5))
    # # model.add(Dropout(.2))


    # # model.add(Bidirectional(LSTM(units, activation='sigmoid'), input_shape=(100, 1)))
    # # model.add(Bidirectional(LSTM(50, activation='sigmoid')))
    # # model.add(TimeDistributed(Dense(1, activation='sigmoid')))
    # model.add(LSTM(50,))
    # model.add(LSTM(25,))
    # # model.add(LSTM(12))
    # model.add(Dense(1))





    model = Sequential()
    model.add(Dense(100, input_shape=(100,)))
    model.add(RBFLayer(100, 0.5))
    model.add(RBFLayer(50, 0.5))
    model.add(Dense(1))








    opt=tf.keras.optimizers.Adam(
    learning_rate=lr,
    name=optim,)
    model.compile(loss='mean_squared_error',optimizer=opt)

    model.summary()


    



    hist=model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=epochs,batch_size=batch,verbose=1)
    print(hist.history)
    model.save("saved_model_single/"+val+"/my_model.h5")
    # model = keras.models.load_model("saved_model_single/"+val+"/my_model.h5")




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
    # y_train2=y_train.reshape(-1,1)
    # y_test2=ytest.reshape(-1,1)

    # print("Y_trsain: "+str(y_train2))
    # print("train_predict: "+str(train_predict))

    ### Train Data RMSE,MSE,MAPE,MAE
    train_err=math.sqrt(mean_squared_error(y_train2,train_predict))
    print("Train RMSE: "+str(train_err))
    print("Train MSE: "+str(mean_squared_error(y_train2,train_predict)))
    print("Train MAPE: "+str(MAPE(y_train2,train_predict)))
    print("Train MAE: "+str(mae(y_train2,train_predict)))



    ### Test Data RMSE,MSE,MAPE,MAE
    test_err=math.sqrt(mean_squared_error(y_test2,test_predict))
    print("Test RMSE: "+str(test_err))
    print("Test MSE: "+str(mean_squared_error(y_test2,test_predict)))
    print("Test MAPE: "+str(MAPE(y_test2,test_predict)))
    print("Test MAE: "+str(mae(y_test2,test_predict)))

    ### Plotting 
    # shift train predictions for plotting
    look_back=100
    trainPredictPlot = np.empty_like(df1)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
    # shift test predictions for plotting
    testPredictPlot = np.empty_like(df1)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
    # plot baseline and predictions
    plt.plot(scaler.inverse_transform(df1))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.title('Model Prediction of Test and Train Data of '+val)
    plt.ylabel('Stock Price')
    plt.xlabel("Time(yr)")
    # plt.xlabel('Time(yr)\nTrain RMSE: '+str(train_err)+" \nTest RMSE: "+str(test_err))
    plt.legend(['Stock Price','Train', 'Test'], loc='upper left')
    plt.savefig('Graphs_Single_LSTM/'+val+' train_test'+'.png')
    plt.cla()
    # plt.show()


    size=len(test_predict)
    plt.plot(scaler.inverse_transform(df1)[-size:])
    plt.plot(test_predict,color='green')
    plt.title('Model Prediction of Test Data '+val)
    plt.ylabel('Stock Price')
    plt.xlabel("Time")
    # plt.xlabel('Time(yr)\nTrain RMSE: '+str(train_err)+" \nTest RMSE: "+str(test_err))
    plt.legend(['Stock Price', 'Test'], loc='upper left')
    plt.savefig('Graphs_Single_LSTM/'+val+' test'+'.png')
    plt.cla()
    # plt.show()

    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('Graphs_Single_LSTM/'+val+' loss'+'.png')
    # plt.show()




# #NEXT 30 days Prediction.
# # len(test_data)

# # x_input=test_data[len(test_data)-100:].reshape(1,-1)
# # x_input.shape





# # temp_input=list(x_input)
# # temp_input=temp_input[0].tolist()

# # temp_input

# # # demonstrate prediction for next 10 days
# # from numpy import array

# # lst_output=[]
# # n_steps=100
# # i=0
# # while(i<30):
    
# #     if(len(temp_input)>100):
# #         #print(temp_input)
# #         x_input=np.array(temp_input[1:])
# #         # print("{} day input {}".format(i,x_input))
# #         x_input=x_input.reshape(1,-1)
# #         x_input = x_input.reshape((1, n_steps, 1))
# #         #print(x_input)
# #         yhat = model.predict(x_input, verbose=0)
# #         # print("{} day output {}".format(i,yhat))
# #         temp_input.extend(yhat[0].tolist())
# #         temp_input=temp_input[1:]
# #         #print(temp_input)
# #         lst_output.extend(yhat.tolist())
# #         i=i+1
# #     else:
# #         x_input = x_input.reshape((1, n_steps,1))
# #         yhat = model.predict(x_input, verbose=0)
# #         # print(yhat[0])
# #         temp_input.extend(yhat[0].tolist())
# #         # print(len(temp_input))
# #         lst_output.extend(yhat.tolist())
# #         i=i+1
    

# # # print(lst_output)

# # day_new=np.arange(1,101)
# # day_pred=np.arange(101,131)

# # import matplotlib.pyplot as plt

# # len(df1)



# # plt.plot(day_new,scaler.inverse_transform(df1[len(df1)-100:]))
# # plt.plot(day_pred,scaler.inverse_transform(lst_output))
# # plt.title('Model accuracy')
# # plt.ylabel('Accuracy')
# # plt.xlabel('Epoch')
# # plt.legend(['Train', 'Test'], loc='upper left')
# # plt.show()

# # df3=df1.tolist()
# # df3.extend(lst_output)
# # plt.plot(df3[2200:])
# # plt.show()


# # df3=scaler.inverse_transform(df3).tolist()

# # plt.plot(df3)
# # plt.show()

