import scipy.io
import sys
import numpy as np
from keras.utils import np_utils
import matplotlib.pyplot as plt
a=scipy.io.loadmat('101.mat')


ecg=a['signal']
ann=a['ann']
ecg=ecg[:,0]
ann=ann[:,0]

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(-1,1))
a1=scaler.fit_transform(ecg)

gl=[0]*650000
for i in ann:
	for j in range(i-50,i+50):
		gl[j]=1

x=[]
y=[]

for i in range(150,650000-150,200):
	x.append(a1[i-150:i+150])
	y.append(gl[i])

x=np.array(x)
y=np.array(y)
#print(x.shape)
#print(y.shape)
#sys.exit(0)
#plt.plot(ecg)
#plt.plot(gl)
#plt.show()

x=x.reshape(3249,1,300)
y=np_utils.to_categorical(y)
y=y.reshape(3249,1,2)


seed=3
np.random.seed(seed)
print('Building....')
from keras.models import Sequential
from keras.layers import Dropout,LSTM,Conv1D,Dense,Activation,Flatten,Embedding,SimpleRNN
model = Sequential()
#model.add(Embedding(300,100))
model.add(SimpleRNN(100,activation='sigmoid',batch_input_shape=(3249, 1, 300),return_sequences=True))
model.add(Dropout(0.2))
model.add(Dense(2,activation='softmax'))
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
print('Training....')
model.fit(x,y,epochs=15,batch_size=9)

a=scipy.io.loadmat('101.mat')
ecg=a['signal']
ann=a['ann']
ecg=ecg[:10000,0]
ecg=scaler.fit_transform(ecg)
gl=[0]*10000

x=[]
y=[]
for i in range(150,10000-150,1):
	x.append(ecg[i-150:i+150])
	y.append(gl[i])

x=np.array(x)
x.shape=(9700,1,300)
t=model.predict(x)
#print(type(t))
t=t.reshape(9700,2)
for l in range(len(t)):
	if(t[l][1]>0.5):
		t[l][1]=1
	else:
		t[l][1]=0
plt.plot(ecg[150:])
plt.plot(t[:,1])
plt.show()
