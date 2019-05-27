#This can be used if you want to re-fit the model


import numpy as np
import pickle
from keras.models import load_model

with open('data.pkl', 'rb') as f:
	x_train, y_train = pickle.load(f)
# name = input("filename: ")
epoch = int(input("epochs: "))
name = "model.h5"
model = load_model(name)
model.fit(x_train, y_train, epochs=epoch)
model.save(name)