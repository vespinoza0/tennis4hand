import csv
import tkinter
from tkinter import filedialog
import os
import datetime
import keras 
import numpy as np
from numpy import genfromtxt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional

root = tkinter.Tk()
root.withdraw()
filez = filedialog.askopenfilenames(parent=root,title='SELECT ALL FEATURE files')
fileList = list(filez)

lens = []   		# list of swing lengths
for i in range(0, len(fileList)):		# go thru all swing files to get max length
	with open(fileList[i]) as csvDataFile:
		csvReader = csv.reader(csvDataFile)
		Hr = list( csv.reader(csvDataFile))
		lens.append(len(Hr))
		
ml = int(max(lens))
numFiles = len(fileList)
output_array = np.zeros((numFiles,ml,3))  # train data is an array of (swingFiles, maxLength, 3)
classes = np.zeros((numFiles,2))
netCount = 0


for i in range(0,len(fileList)):
	Hr= genfromtxt(fileList[i], delimiter=',')
	result = np.zeros((ml,3))
	data = Hr[:,0:3]
	ting = Hr[0,4:6]
	if 10 in ting:  # 10 means net
		#print(ting)
		netCount=netCount+1
		continue
	else:
		classes[i,:] = ting
		result[:data.shape[0],:data.shape[1]] = data
		output_array[i] = result

print("This is the netcount",netCount)
#output_array2 = np.zeros((numFiles-netCount,ml,3))
goodshots = numFiles -netCount 
newArray = output_array[0:goodshots,:,:]
newY = classes[0:goodshots,:]
print("Good shots array is ", newArray.shape)
print("New Y train shape is", newY.shape)

y_train = np.random.randint(2, size=(numFiles, 2)) 
y_train2 = np.random.randint(10, size=(numFiles, 2))
y_train2 = classes

# Convert labels to categorical one-hot encoding
ylabels = keras.utils.to_categorical(y_train2, num_classes=10)


x_train = output_array
print("the shape of x_train array: ",x_train.shape)
print("the shape of y_train array: ",y_train.shape)
print("and the shape of new y train is:", classes.shape)
input0 = x_train[0]

# Generate dummy validation data
x_val = np.random.random((100, ml, 3))
y_val = np.random.random((100, 2))

max_features = 20000
# cut texts after this number of words
# (among top max_features most common words)
# maxlen = 100
batch_size = 2		  
batch_size=1
	
model = Sequential()	
model.add(LSTM(32, return_sequences=True, stateful=True,
               batch_input_shape=(batch_size, ml, 3)))
model.add(LSTM(32, return_sequences=True, stateful=True))
model.add(LSTM(32, stateful=True))
#model.add(Dense(2, activation='sigmoid'))
model.add(Dense(2, activation='tanh'))

# try sgd optmizer maybe
model.compile(loss='categorical_crossentropy',
				optimizer='rmsprop',
				metrics=['accuracy'])
			  
model.summary()
			  
print('Train...')			  
#model.fit(x_train, y_train, batch_size=1, epochs=10)
model.fit(x_train, y_train2, batch_size=1, epochs=5)



# model.fit(x_train, y_train,
          # batch_size=batch_size,
          # epochs=4, validation_data=(x_val, y_val))
		  
		  
