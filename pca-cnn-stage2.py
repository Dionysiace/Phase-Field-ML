

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

This script use pca reuslts to train cnn network
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import pickle

def train2():
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import Dropout
    from keras.layers import Flatten
    from keras.layers import Conv2D
    from keras.layers import MaxPooling2D
    import matplotlib.pyplot as plt
    def load_csv(path):
        data_read = pd.read_csv(path,header=None,dtype=object)
        list = data_read.values.tolist()
        data = np.array(list)
        # print(data)
        return data
    
    def dataset_gen(window_start,window_width,sample_start,sample_end):
        dpath="./download-0/"
        dtail1="_ppt.csv"
        dtail2="_chara.txt"
        data_1d=np.zeros((16*window_width,1));
        mat_add=np.empty([1,16,window_width])
        y_train=np.empty([0,2])
        for i in range(sample_start,sample_end):
            mat=load_csv(dpath+str(i)+dtail2)[:2,0].reshape([1,2]).astype('float64')
            if mat[0,0]<0:
                continue
            y_train=np.append(y_train,mat,axis=0)
            
            mat=load_csv(dpath+str(i)+dtail1)
            mat=mat[:,window_start:window_start+window_width].astype('float64')
            #mat=np.multiply(np.ones([16,window_width])*0.75,mat>=0.15)+0.25
            mat=mat.reshape(16*window_width,1)
            data_1d=np.hstack([data_1d,mat])
            
        data_1d=data_1d[:,1:].astype('float64')
        mean=np.empty([0,1])
        for i in range(len(data_1d[:,0])):
            mean=np.vstack([mean,np.ones((1,1))*np.sum(data_1d[i,:])/len(data_1d[0,:])])
            data_1d[i,:]=data_1d[i,:]-np.ones((1,len(data_1d[0,:])))*np.sum(data_1d[i,:])/len(data_1d[0,:])
            
        x_train=data_1d
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
      #get new data
      data=np.dot(feature,X)
      return data,feature
            
    def pca_recon(data,feature,mean,window_width):
        data_1d=np.dot(np.transpose(feature),data)+mean
        data_recon=data_1d.reshape(16,window_width)
        return data_recon
    
    import keras.backend as K
    from keras.callbacks import LearningRateScheduler
     
    def scheduler(epoch):
        
        if epoch % 200 == 0 and epoch != 0:
            lr = K.get_value(model.optimizer.lr)
            K.set_value(model.optimizer.lr, lr * 0.2)
            #print("lr changed to {}".format(lr * 0.2))
        return K.get_value(model.optimizer.lr)
    
    
    x_all,y_all,mean=dataset_gen(10,32,0,2000)
    
    pca_result,feature=pca(x_all,3)
    x_all=np.transpose(pca_result)
    
    test_inp=np.empty([0,16,32])
    test_add=np.empty([1,16,32])
    for i in range(len(x_all)):
        test_add[0,:,:]=pca_recon(np.transpose(x_all[i,:]),feature,mean,32)
        test_inp=np.append(test_inp,test_add,axis=0)
    
    x_all=test_inp
    # shuffled index import
    with open('./index.pickle','rb') as f:
        param=pickle.load(f)
    index=param[0]
    # do not shuffle in k-fold
    # np.random.shuffle((index))
    
    scaler=MinMaxScaler(feature_range=(0,1))
    y_all=scaler.fit_transform(y_all)
    
    
    n_sample=int(len(x_all)*0.8)
    x_train=x_all[np.concatenate((index[0:int(n_sample*0)],index[int(n_sample*0.2):n_sample])),:]
    y_train=y_all[np.concatenate((index[0:int(n_sample*0)],index[int(n_sample*0.2):n_sample])),:]
    x_test=x_all[index[int(n_sample*0):int(n_sample*0.2)],:]
    y_test=y_all[index[int(n_sample*0):int(n_sample*0.2)],:]
    
    
    
    
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', input_shape=(16, 32, 1),
                     activation='relu'))
    
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Dropout(0.2))
    
    model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    
    model.add(Flatten())
    
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='linear'))
    model.add(Dense(2))
    
    model.compile(loss='mse', optimizer='adam')
    
    reduce_lr = LearningRateScheduler(scheduler)
    history=model.fit(x_train, y_train, validation_data=(x_test, y_test),
              epochs=50, batch_size=32, callbacks=[reduce_lr])
    
    scores = model.evaluate(x_test, y_test, verbose=0)
    print(scores)
    
    test=model.predict(x_all)
    plt.figure()
    plt.subplot(1,2,1)
    plt.scatter(test[:,0],y_all[:,0],c='red',s=4)
    plt.plot(np.array([0,1]))
    plt.axis('equal')
    plt.subplot(1,2,2)
    plt.scatter(test[:,1],y_all[:,1],c='black',s=4)
    plt.plot(np.array([0,1]))
    plt.axis('equal')
    
    epochs=range(len(history.history['loss']))
    plt.figure()
    plt.plot(epochs,history.history['loss'],'b',label='Training loss')
    plt.plot(epochs,history.history['val_loss'],'r',label='Validation val_loss')
    plt.legend()
    plt.yscale("log")
    
    return model,scaler


def load_csv(path):
    data_read = pd.read_csv(path,header=None,dtype=object)
    list = data_read.values.tolist()
    data = np.array(list)
    # print(data)
    return data

def input_data_gen(sample_start,sample_end):
    dpath="./download-0/"
    dtail1="_ppt.csv"
    dtail2="_chara.txt"
    x_train=np.empty([0,8])
    y_train=np.empty([0,2])
    mat_add=np.empty([1,16,32])
    profile=np.empty([0,16,32])
    for i in range(sample_start,sample_end):
        mat=load_csv(dpath+str(i)+dtail2)[:2,0].reshape([1,2]).astype('float64')
        if mat[0,0]<0:
            continue
        y_train=np.append(y_train,mat,axis=0)
        mat=load_csv(dpath+str(i)+dtail2)[9:17,0].reshape([1,8]).astype('float64')
        x_train=np.append(x_train,mat,axis=0)
        
        mat=load_csv(dpath+str(i)+dtail1)
        mat_add[0,:,:]=mat[:,10:10+32].astype('float64')
        #mat_add[0,:,:]=np.multiply(np.ones([16,window_width])*0.75,mat_add[0,:,:]>=0.4)+0.25
        profile=np.append(profile,mat_add,axis=0)
    
    return x_train,y_train,profile

def pca_recon(data,feature,mean,window_width):
    data_1d=np.dot(np.transpose(feature),data)+mean
    data_recon=data_1d.reshape(16,window_width)
    return data_recon

# model1,feature,mean,scaler=train1()

with open('./model1.pickle','rb') as f:
    param=pickle.load(f)
feature=param[0]
mean=param[1]
scaler=param[2]
from keras.models import load_model
model1 = load_model('model1.h5')

model2,scaler2=train2()


x_all,y_all,profile=input_data_gen(1900,2000)
n_sample=len(x_all)


temp=model1.predict(x_all[int(n_sample*0):,:])
temp=scaler.inverse_transform(temp)

test_inp=np.empty([0,16,32])
test_add=np.empty([1,16,32])
for i in range(len(temp)):
    test_add[0,:,:]=pca_recon(np.transpose(temp[i,:]),feature,mean,32)
    test_inp=np.append(test_inp,test_add,axis=0)



test=model2.predict(test_inp)
y_all=scaler2.fit_transform(y_all)

mse1=np.sum((test[:,0]-y_all[int(n_sample*0):,0])**2)/len(test)
mse2=np.sum((test[:,1]-y_all[int(n_sample*0):,1])**2)/len(test)

# results evaluation
plt.figure(figsize=(7,3))
plt.subplot(1,2,1)
plt.scatter(test[:,0],y_all[int(n_sample*0):,0],c='red',s=4)
# plt.plot(np.array([1,1.2]),np.array([1,1.2]))
plt.plot(np.array([0,1]),np.array([0,1]))
#plt.axis('equal')
plt.subplot(1,2,2)
plt.scatter(test[:,1],y_all[int(n_sample*0):,1],c='black',s=4)
# plt.plot(np.array([2,5]),np.array([2,5]))
plt.plot(np.array([0,1]),np.array([0,1]))

model2.save('model2.h5')
with open('./model.pickle','wb') as f:
    pickle.dump([feature,mean,scaler,scaler2],f)
