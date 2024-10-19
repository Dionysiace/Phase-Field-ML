# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt	
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dropout
import pickle

def load_csv(path):
    data_read = pd.read_csv(path,header=None,dtype=object)
    list = data_read.values.tolist()
    data = np.array(list)
    # print(data)
    return data

def input_data_gen(window_start,window_width,sample_start,sample_end):
    dpath="./download-0/"
    dtail1="_ppt.csv"
    dtail2="_chara.txt"
    data_1d=np.zeros((16*window_width,1));
    x_train=np.empty([0,8])
    for i in range(sample_start,sample_end):
        mat=load_csv(dpath+str(i)+dtail2)[:2,0].reshape([1,2]).astype('float64')
        if mat[0,0]<0:
            continue
        mat=load_csv(dpath+str(i)+dtail1)
        mat=mat[0:16,window_start:window_start+window_width].astype('float64')
        # mat=np.multiply(np.ones([16,window_width])*0.75,mat>=0.4)+0.25
        
        
        mat=mat.reshape(16*window_width,1)
        data_1d=np.hstack([data_1d,mat])
        mat=load_csv(dpath+str(i)+dtail2)[9:17,0].reshape([1,8]).astype('float64')
        x_train=np.append(x_train,mat,axis=0)
    data_1d=data_1d[:,1:].astype('float64')
    mean=np.empty([0,1])
    for i in range(len(data_1d[:,0])):
        mean=np.vstack([mean,np.ones((1,1))*np.sum(data_1d[i,:])/len(data_1d[0,:])])
        data_1d[i,:]=data_1d[i,:]-np.ones((1,len(data_1d[0,:])))*np.sum(data_1d[i,:])/len(data_1d[0,:])
        
    y_train=data_1d
    return x_train,y_train,mean[:,0]

def pca(X,k):#k is the components you want
  #mean of each feature
  n_features,n_samples= X.shape
  scatter_matrix=np.dot(X,np.transpose(X))
  #Calculate the eigenvectors and eigenvalues
  eig_val, eig_vec = np.linalg.eig(scatter_matrix)
  #eig_vec=np.dot(X,eig_vec)
  eig_pairs = [(np.abs(eig_val[i]), np.real(eig_vec[:,i])) for i in range(n_features)]
  # sort eig_vec based on eig_val from highest to lowest
  eig_pairs.sort(reverse=True,key=lambda x:x[0])
  # select the top k eig_vec
  feature=np.array([ele[1] for ele in eig_pairs[:k]])
  eig_val=np.array([ele[0] for ele in eig_pairs])
  #get new data
  data=np.dot(feature,X)
  return data,eig_val,feature
        
import keras.backend as K
from keras.callbacks import LearningRateScheduler
 
def scheduler(epoch):
    # 每隔100个epoch，学习率减小为原来的1/10
    if epoch % 120 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.2)
        print("lr changed to {}".format(lr * 0.2))
    return K.get_value(model.optimizer.lr)


x_all,y_all,mean=input_data_gen(10,32,0,2000)

pca_result,eig_val,feature=pca(y_all,3)
y_all=np.transpose(pca_result)

# shuffled index import
with open('./index.pickle','rb') as f:
    param=pickle.load(f)
index=param[0]
# index=np.arange(len(x_all))
# np.random.shuffle((index))

scaler=MinMaxScaler(feature_range=(0,1))
y_all=scaler.fit_transform(y_all)


n_sample=int(len(x_all)*0.8)
# x_train=x_all[index[0:int(n_sample*0.8)],:]
# y_train=y_all[index[0:int(n_sample*0.8)],:]
# x_test=x_all[index[int(n_sample*0.8):int(n_sample*1)],:]
# y_test=y_all[index[int(n_sample*0.8):int(n_sample*1)],:]

# k-fold splitting
x_train=x_all[np.concatenate((index[0:int(n_sample*0)],index[int(n_sample*0.2):n_sample])),:]
y_train=y_all[np.concatenate((index[0:int(n_sample*0)],index[int(n_sample*0.2):n_sample])),:]
x_test=x_all[index[int(n_sample*0):int(n_sample*0.2)],:]
y_test=y_all[index[int(n_sample*0):int(n_sample*0.2)],:]


model = Sequential()
model.add(Dense(80, input_dim = 8, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(8, activation='linear'))
model.add(Dense(3))
model.compile(loss='mse', optimizer='adam')

reduce_lr = LearningRateScheduler(scheduler)
history=model.fit(x_train, y_train, validation_data=(x_test, y_test),
          epochs=200, batch_size=12, callbacks=[reduce_lr])
#history=model.fit(x_train, y_train, validation_data=(x_test, y_test),
#          epochs=200, batch_size=35)

epochs=range(len(history.history['loss']))
plt.figure()
plt.plot(epochs,history.history['loss'],'b',label='Training loss')
plt.plot(epochs,history.history['val_loss'],'r',label='Validation val_loss')
plt.legend()
plt.yscale("log")

plt.figure()
eig_val_sum=np.empty([len(eig_val),1])
for i in range(len(eig_val)):
    eig_val_sum[i]=sum(eig_val[:i])/sum(eig_val)
plt.plot(eig_val_sum[:20])

#results evaluation
test=model.predict(x_all)
plt.figure()
plt.subplot(2,2,1)
plt.scatter(test[:,0],y_all[:,0],c='red',s=4)
plt.axis('equal')
plt.plot(np.array([0,1]))
plt.subplot(2,2,2)
plt.scatter(test[:,1],y_all[:,1],c='black',s=4)
plt.axis('equal')
plt.plot(np.array([0,1]))
plt.subplot(2,2,3)
plt.scatter(test[:,2],y_all[:,2],c='black',s=4)
plt.axis('equal')
plt.plot(np.array([0,1]))


model.save('model1.h5')
with open('./model1.pickle','wb') as f:
    pickle.dump([feature,mean,scaler,index],f)
