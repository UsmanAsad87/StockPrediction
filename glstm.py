import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split as split

from keras.layers import LSTM, Input, Dense
from keras.models import Model

from deap import base, creator, tools, algorithms
from scipy.stats import bernoulli
from bitstring import BitArray

def prepare_dataset(data, window_size):
    window_size=1
    print(data.shape)
    X, Y = np.empty((window_size,7)), np.empty((7))
    print(X)
    print(X.shape)
    print(data[1])
    for i in range(len(data)-window_size-1):
        X = np.vstack([X,data[i:(i + window_size)]])
        Y = np.vstack([Y,data[i + window_size]])
        
    print(X.shape)

    print(Y.shape)
    X = np.reshape(X,(len(X),window_size,7))
    Y = np.reshape(Y,(len(Y),7))
    print(X.shape)
    print(Y.shape)
    return X, Y

def train_evaluate(ga_individual_solution):   
    # Decode GA solution to integer for window_size and num_units
    window_size_bits = BitArray(ga_individual_solution[0:6])
    num_units_bits = BitArray(ga_individual_solution[6:]) 
    window_size = window_size_bits.uint
    num_units = num_units_bits.uint
    print('\nWindow Size: ', window_size, ', Num of Units: ', num_units)
    
    # Return fitness score of 100 if window_size or num_unit is zero
    if window_size == 0 or num_units == 0:
        return 100, 
    print(train_data)
    print(train_data.shape)
    # Segment the train_data based on new window_size; split into train and validation (80/20)
    X,Y = prepare_dataset(train_data,1)
    X_train, X_val, y_train, y_val = split(X, Y, test_size = 0.20, random_state = 1120)
    print("done")
    # Train LSTM model and predict on validation set
    inputs = Input(shape=(1,7))
    x = LSTM(num_units, input_shape=(1,7))(inputs)
    predictions = Dense(7, activation='linear')(x)
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='adam',loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=1, batch_size=10,shuffle=True)
    y_pred = model.predict(X_val)
    
    # Calculate the RMSE score as fitness score for GA
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    print('Validation RMSE: ', rmse,'\n')
    
    return rmse,

np.random.seed(1120)
col_Names=["date", "u", "v", "s","d","u", "v", "s","d","u", "v", "s","d","u", "v", "s","d","pwr"]
data = pd.read_csv('E:/sep-2018/GA-LSTM/code/WindFarm_2.csv',names=col_Names)


values = data.values
# integer encode direction
# drop columns we don't want to predict
data.drop(data.columns[[0]], axis=1, inplace=True)
data = np.reshape(np.array(data.values),(len(data['wp1']),7))


print(data.shape)
# Use first 17,257 points as training/validation and rest of the 1500 points as test set.
train_data = data[0:17257]
test_data = data[17257:]

print(test_data)

population_size = 1
num_generations = 1
gene_length = 10

# As we are trying to minimize the RMSE score, that's why using -1.0. 
# In case, when you want to maximize accuracy for instance, use 1.0
creator.create('FitnessMax', base.Fitness, weights = (-1.0,))
creator.create('Individual', list , fitness = creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register('binary', bernoulli.rvs, 0.5)
toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.binary, 
n = gene_length)
toolbox.register('population', tools.initRepeat, list , toolbox.individual)

toolbox.register('mate', tools.cxOrdered)
toolbox.register('mutate', tools.mutShuffleIndexes, indpb = 0.6)
toolbox.register('select', tools.selRoulette)
toolbox.register('evaluate', train_evaluate)

population = toolbox.population(n = population_size)
r = algorithms.eaSimple(population, toolbox, cxpb = 0.4, mutpb = 0.1, 
ngen = num_generations, verbose = False)

best_individuals = tools.selBest(population,k = 1)
best_window_size = None
best_num_units = None

for bi in best_individuals:
    window_size_bits = BitArray(bi[0:6])
    num_units_bits = BitArray(bi[6:])
    best_window_size = window_size_bits.uint
    best_num_units = num_units_bits.uint
    print('\nWindow Size: ', best_window_size, ', Num of Units: ', best_num_units)
best_window_size=1

X_train,y_train = prepare_dataset(train_data,1)
X_test, y_test = prepare_dataset(test_data,1)

inputs = Input(shape=(best_window_size,7))
x = LSTM(best_num_units, input_shape=(best_window_size,7))(inputs)
predictions = Dense(7, activation='linear')(x)
model = Model(inputs = inputs, outputs = predictions)
model.compile(optimizer='adam',loss='mean_squared_error')
model.fit(X_train, y_train, epochs=1, batch_size=10,shuffle=True)
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print('Test RMSE: ', rmse)
