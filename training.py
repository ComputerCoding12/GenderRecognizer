
#Importing the necessary libraries
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Flatten, SimpleRNN
import pickle


#Accessing the data that we have processed in the preprocessing file
with open('data.pkl', 'rb') as f:
	x, y = pickle.load(f)
x_train, x_test = [x[:int(len(x) * (0.8))], x[int(len(x) * (0.8)):]]
y_train, y_test = [y[:int(len(y) * (0.8))], y[int(len(y) * (0.8)):]]


#Creating a sequential model
model = Sequential()
model.add(LSTM(x_train.shape[-1], input_shape=[x_train.shape[-2], x_train.shape[-1]], return_sequences=True))
model.add(LSTM(40, return_sequences=True))
model.add(LSTM(40, return_sequences=True))
model.add(LSTM(40, return_sequences=True))
model.add(Flatten())
model.add(Dense(10))
model.add(Dense(2))
model.compile(optimizer="adam", loss='mse')
model.fit(x_train, y_train, epochs=10)
model.save("model.h5")
