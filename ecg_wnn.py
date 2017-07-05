import scipy.io
import numpy as np
import math
from keras.utils import np_utils
import matplotlib.pyplot as plt
#a=scipy.io.loadmat('mgh001.mat')
import sys

from keras.models import Sequential
from keras.layers import LSTM,Conv2D,MaxPooling2D,Dense,Activation,Flatten
model=Sequential()
#model.add(LSTM(2,activation='softmax',input_shape=(1,150)))
#model.add(Embedding(max_features,embedding_dims,input_length=maxlen))
model.add(Conv2D(1,(2,30),activation='sigmoid',input_shape=(2,300,1)))
#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(2,activation='softmax'))
model.summary()
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

for num in range(10,100):
	bch='mgh0'+str(num)+'.mat'
	a=scipy.io.loadmat(bch)
	ecg=a['signal']
	ann=a['ann']
	ecg=ecg[:150000,0:2]
	ann=ann[:,0]
	gl=[0]*150000
	nop=0
	for i in ecg:
		for j in i:
			if(math.isnan(float(j))):
				nop=1
	if(nop==1):
		continue	
	from sklearn.preprocessing import MinMaxScaler
	scaler=MinMaxScaler(feature_range=(-1,1))
	ecg=scaler.fit_transform(ecg)
	
	for i in ann:
		if(i<149960):
			for j in range(i-40,i+40):
				gl[j]=1

	x=[]
	y=[]
	for i in range(150,150000-150,200):
		x.append(ecg[i-150:i+150])
		y.append(gl[i])

	x=np.array(x)
	y=np.array(y)
	#print(x.shape)
	#plt.plot(y)
	#plt.show()
	x=x.reshape(749,2,300,1)
	y=np_utils.to_categorical(y)
	y=y.reshape(749,2)
	#print(y[1:100])
	print("Signal "+str(num))
	model.fit(x,y,epochs=75,batch_size=35)


a=scipy.io.loadmat('mgh011.mat')
ecg=a['signal']
ann=a['ann']
ecg=ecg[:10000,0:2]
ann=ann[:,0]
gl=[0]*10000

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(-1,1))
ecg=scaler.fit_transform(ecg)


x=[]
y=[]
for i in range(150,10000-150,1):
	x.append(ecg[i-150:i+150])
	y.append(gl[i])

x=np.array(x)
x.shape=(9700,2,300,1)
t=model.predict(x)
#print(type(t))
for l in range(len(t)):
	if(t[l][0]>0.5):
		t[l][0]=0
	else:
		t[l][0]=1
plt.plot(ecg[150:,0])
plt.plot(t[:,0])
plt.show()
