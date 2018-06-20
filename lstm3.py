import csv
import tkinter
from tkinter import filedialog
import os
import datetime
import keras 
import numpy as np
from numpy import genfromtxt
from keras.models import Sequential
from keras.callbacks import TensorBoard
from keras.layers import Dense, Dropout, Embedding, Bidirectional
from keras.layers import Embedding, LSTM
from keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt
import random
from sklearn.metrics import classification_report, confusion_matrix

def getMaxSwinglengthCSV(fileList):
	lens=[]
	nets =0
	for i in range(0, len(fileList)):		# go thru all swing files to get max length
		with open(fileList[i]) as csvDataFile:
			csvReader = csv.reader(csvDataFile)
			Hr = list(csv.reader(csvDataFile))
			lens.append(len(Hr))
			Hr= genfromtxt(fileList[i], delimiter=',')
			data = Hr[:,0:3]
			t = np.sum(data,axis=0)
			tt = np.sum(t)
			if tt==0:
				print("this is a ZERO file!!",fileList[i])
	lll = np.array(lens)
	print("avg seq length mean is ",lll.mean(axis=0))
	return(max(lens))
	
def mapSwingsCategorical(fileList, ml):  # and filter nets 
	numFiles =len(fileList)
	output_array = np.zeros((numFiles,ml,3))  # train data is an array of (swingFiles, maxLength,features)
	bigArr = np.zeros((numFiles*ml,3))
	Xtar = []
	netCount = 0
	ii=0
	j = 0 
	lengths = []
	for i in range(0,len(fileList)):
		with open(fileList[i]) as csvDataFile:
			csvReader = csv.reader(csvDataFile)
			Hr = list(csv.reader(csvDataFile))
		result = np.zeros((ml,3)) 					# place holder for variable length data
		ting = Hr[0][4:6]
		x = ting[0]
		y = ting[1]
		
	
		if x == "10" or x == "NaN":  		# 10 or 9 means net
			netCount=netCount+1
			continue
		else:
			Hr= genfromtxt(fileList[i], delimiter=',')
			data = Hr[:,0:3]
			lengths.append(len(data))
			result[:data.shape[0],:data.shape[1]] = data # put the data
			output_array[j,:,:] = result
			j = j+1
			bigArr[ii:ii+data.shape[0],:] = data
			ii = ii +data.shape[0]
			if x == '-1':
				Xtar.append(0)
				continue
			elif x == '-0.5':
				Xtar.append(1)
				continue
			elif x == '0':
				Xtar.append(2)
				continue
			elif x == '0.5':
				Xtar.append(3)
				continue
			elif x == '1':
				Xtar.append(4)
				continue
			else:
				print("WHOAA WHAT IS THIS!", x)
	
	Xtar = np.array(Xtar)
	goodSwings = numFiles-netCount
	bigArr = bigArr[:ii,:]
	avg = bigArr.mean(axis=0)
	std = bigArr.std(axis=0)
	output_array=output_array[:goodSwings,:,:]
	print("output array shape mapSwingFunc ",output_array.shape)
	ass = checkZeros(output_array)
	if ass >0:
		print("bug is in map swings!")

	return output_array, Xtar, bigArr
	
def normFeature(fileList,mean,std):	 ## go thru data, subtract each feature its mean and divide by  
	numFiles = len(fileList)
	output_array = np.zeros((numFiles,ml,3))
	netCount = 0
	ii=0
	j=0
	bigArr = np.zeros((numFiles*ml,3))
	print("normalizing now!")
	for i in range(numFiles):
		with open(fileList[i]) as csvDataFile:
			csvReader = csv.reader(csvDataFile)
			Hr = list(csv.reader(csvDataFile))
		result = np.zeros((ml,3)) 					# place holder for variable length data
		ting = Hr[0][4:6]
		x = ting[0]
		y = ting[1]
		Hr= genfromtxt(fileList[i], delimiter=',')
		data = Hr[:,0:3]
		
		if x == "10" or x == "NaN":  		# 10 or 9 means net
			netCount=netCount+1
			continue
		else:
			data =data-mean
			data = data/std
			bigArr[ii:ii+data.shape[0],:] = data
			ii = ii + data.shape[0]
			result[:data.shape[0],:data.shape[1]] = data # put the data
			output_array[j,:,:] = result
			j = j+1
			
	goodSwings = numFiles-netCount
	print("ii", ii)
	bigArr = bigArr[:ii,:]
	output_array=output_array[:goodSwings,:,:]
	return output_array, bigArr
	

def shuffleData(x, y):		# shuffles data, x = seqs, y =targets 
	goodSwings = y.shape[0]
	print("NOW shuffling!!")
	if x.shape[0] == y.shape[0]: 
		x_1 = np.zeros(x.shape)
		y_1 = np.zeros(y.shape)
		r = list(range(goodSwings))
		random.shuffle(r)
		for i in range(len(r)):
			x_1[i,:,:] = x[r[i],:,:]
			y_1[i] = y[r[i]]
			t = np.sum(x[i,:,:],axis=0)
			tt = np.sum(t)
		return x_1, y_1
		
def checkZeros(x):
	count = 0 
	for i in range(x.shape[0]):
		seq = x[i,:,:]
		t = np.sum(seq,axis=0)
		tt = np.sum(t)
		if tt == 0:
			count = count+1
	return count
	
def shuffle(x,y):
	r = list(range(x.shape[0]))
	print("r range", r[-1])
	random.shuffle(r)
	x_1 = np.zeros(x.shape)
	y_1 = np.zeros(y.shape)
	for i in range(len(r)):
		x_1[i,:,:] = x[r[i],:,:]
		y_1[i] = y[r[i]]
	return x_1, y_1
	
def addNoise(y):
	l = y.shape[0]
	s = np.random.normal(0, 1, l)
	return y+s
	
def plotHisory(epochs,loss,val_loss):
	epochs = range(1,epochs+1)		  
	plt.figure()
	plt.plot(epochs,loss,'bo',label= 'Training acc')
	plt.plot(epochs,val_loss,'b',label= 'Validation acc')
	plt.title('Training and validiation Accuracy')
	plt.legend()
	plt.show()
	
# main stuff
if __name__ == "__main__":
	root = tkinter.Tk()
	root.withdraw()
	filez = filedialog.askopenfilenames(parent=root,title='SELECT ALL FEATURE files')
	fileList = list(filez)

	print("total swing files", len(fileList))
	ml = getMaxSwinglengthCSV(fileList)

	x,y,bigArr =  mapSwingsCategorical(fileList, ml)  # returns seqs, targets, and long arr of features
	print("this is the new func to filter nets!")
	print(x.shape)
	print(y.shape)
	mean = bigArr.mean(axis=0)
	std = bigArr.std(axis=0)

	#plt.hist(bigArr) plt.show
	aa = checkZeros(x)
	if aa >0:
		print("found zeros after mapSwings ",aa)
	
	
	normX, bigArr2 = normFeature(fileList ,mean, std)
	print("new norm x shape is", normX.shape)
	aa = checkZeros(normX)
	if aa >0:
		print("found zeros after norm ",aa)
	Xtrain = normX
	
	xx, yy = shuffleData(Xtrain,y)
	
	aa = checkZeros(xx)
	if aa >0:
		print("found zeros after shuffle data", aa)
		
	xx, yy = shuffle(Xtrain,y)
	aa = checkZeros(xx)
	if aa > 0:
		print("found zeros after new shuffle", aa)
	else:
		print("good shuffle!!")
	
	print("shuffled x shape is", xx.shape)
	zeros = 0
	
	
## training data
	timeSteps =ml
	co= Xtrain.shape[0]%10
	co = Xtrain.shape[0]-co
	Xtrain= xx[:co,:,:] 
	Xtar = yy[:co]
	# plt.hist(Xtar)
	# plt.show()
	
	Xtar = keras.utils.to_categorical(Xtar)
	# plt.hist(bigArr2)
	# plt.show()


## test data stuff
	y_testOg= Xtar
	batch =5
	eps = 10
	end = 1-0.2241379 
	s = round(Xtrain.shape[0]*end)
	print("this is the 77 mark", s)
	x_test = Xtrain[s:,:,:]
	print("test shape is ",x_test.shape)
	y_test = Xtar[s:]
	print("test target shape is ",y_test.shape)
	y_testOg = y_testOg[s:] 
	print("test target OG shape is ",y_testOg.shape)

## model stuff
	model = Sequential()	
	model.add(LSTM(32, return_sequences=True, stateful=True, dropout=0.3, recurrent_dropout=0.3,
               batch_input_shape=(batch, timeSteps, 3)))
	model.add(LSTM(32, return_sequences=True, stateful=True, dropout=0.1,recurrent_dropout=0.1))
	model.add(LSTM(16, stateful=True, activation = 'relu'))
	model.add(Dense(5, activation='softmax'))

	model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
######################################################################
	history = model.fit(Xtrain, Xtar,
		epochs=eps,
		batch_size=batch, validation_split=0.2413793)
	model.save('lstm3_model.h5')
	print("#############################################################################")
	#print(y_testOg)
	print("#############################################################################")
	#y_predict = np.argmax(y_predict, axis=1)
	#print(y_predict)
	print("#############################################################################")
	
	

	y_predict = model.predict_classes(x_test,batch_size=batch)
	print(y_predict)
	print("######################################################################")
	loss = history.history['acc']
	val_loss = history.history['val_acc']
	

	
	p = model.predict_proba(x_test, batch_size = batch)

	target_names= ['0','1','2','3','4']
	print(classification_report(np.argmax(y_test,axis=1), y_predict))#, target_names=target_names))
	print(confusion_matrix(np.argmax(y_test,axis=1), y_predict))
	
	plotHisory(eps,loss,val_loss)
	




	
	
	
	
	
