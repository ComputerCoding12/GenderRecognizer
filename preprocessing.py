
#Importing necessary modules
import numpy as np
import pickle
import pandas as pd
import re


#Setting a Pattern For Regular Expression
pattern = r"[^A-Z^a-z]"

#The Text Embedding is loaded from the embedding.pkl file
with open('embedding.pkl', 'rb') as f:
	ref = pickle.load(f)

#The data from the csv files is extracted
female = pd.read_csv('dataset/Indian-Female-Names.csv').iloc[:,:2]
male = pd.read_csv("dataset/Indian-Male-Names.csv").iloc[:,:2]

#The data is merged using the append function
data = female.append(male)
np.random.shuffle(data.values)

def classifier(string):
#The Expected output for the given data is precessed using the function
	if string == 'f':
		return np.array([0,1])
	elif string == 'm':
		return np.array([1,0])

data = pd.concat([data['name'], data['gender'].apply(classifier)], axis=1)
data.dropna()
data.index = range(0, data.shape[0])

rows_to_drop = []


for i in range(data.shape[0]):
	dat = data.iloc[i,0]
	if type(dat) != str:
		rows_to_drop.append(i)
		continue
	if re.findall(pattern, dat) != []:
		rows_to_drop.append(i)

data = data.drop(rows_to_drop)
names = data['name'].values
labels = data['gender'].values
max_len = 19
processed_names = []
for i in names:
	datas = np.zeros([19, 71])
	for j in range(len(i)):
		datas[j, ref['token'][i[j]]] = 1
	processed_names.append(datas)
processed_labels = []
for i in labels:
	processed_labels.append(i)
    
    
#Here The training data is stored into a data.pkl file
with open('data.pkl', 'wb') as f:
	pickle.dump([np.array(processed_names), np.array(processed_labels)], f)