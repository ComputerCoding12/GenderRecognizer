import numpy as np
from keras.models import load_model
import pickle
#Accessing the embedding file that is used
with open("embedding.pkl", 'rb') as f:
	ref = pickle.load(f)['token']
# name = input("filename: ")
name = "model.h5"
model = load_model(name)
done = False

#Using a dummy function the produce a revalent prediction
def _(inp):
	if np.argmax(inp) == 0:
		return "male"
	else:
		return "female"
while not done:
	data = input(": ")
	if data == "quit":
		done = True
		continue
	ind = np.zeros([1, 19, 71])
	for i in range(len(data)):
		ind[0, i, ref[data[i]]] = 1
	out = model.predict(ind)[0]
	print(f": {_(out)} \nMale: {round(out[0] * 100, 2)}% \nFemale: {round(out[-1] * 100, 2)}%")
