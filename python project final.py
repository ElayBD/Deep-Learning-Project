# -*- coding: utf-8 -*-
"""
Created on Mon May 17 11:57:51 2021

@author: PC
"""
import files
import random
from PIL import Image
from numpy import array
import numpy as np
import os
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Activation, MaxPooling2D, Dropout


def to_matrix(path):
    """

    Parameters
    ----------
    path : string
        DESCRIPTION: a path to the dataset

    Returns
    -------
    images : np.array
        DESCRIPTION: an array of the matrixs of the images
    labels : list
        DESCRIPTION: a list of labels for every picture 

    """
    #newpath= os.path.join(path, "train")
    images= []
    labels= [] # 0- adult, 1- kid
    counter= 0
    for subdir in os.listdir(path):
        imglist= os.listdir(os.path.join(path, subdir))
        for img in imglist:
            # access to a specific image- 4u
            im= Image.open(os.path.join(path, subdir, img))
            arr_im= array(im)
            images.append(arr_im)
            labels.append(counter)
        counter+= 1
    return images, labels



def test(res,labels2):
    """
    

    Parameters
    ----------
    res : np array
        DESCRIPTION: an array that includes the predictions of the model
    labels2 : list
        DESCRIPTION: a list of labels for every picture

    Returns
    -------
    test_accuracy : floawt
        DESCRIPTION: the accuracy of the predictions of model 

    """
    res= res.reshape(len(res))
    error=0
    
    for i in range(len(res)):
        if i in range(int(len(res)/2)):
            if res[i]>0.5:
                error+=1
        else:
            if res[i]<0.5:
                error+=1
    test_accuracy=(len(labels2)-error)/(len(labels2))*100
    return test_accuracy
        


print("please type a name for a dataset folder")
folder_name=input()
files.resize("E:/Dataset/adult")
files.resize("E:/Dataset/kid")
path=os.path.join("E:/",folder_name)
try:
    os.mkdir("E:/"+folder_name)
except:
    print("this directory is already exist")
path_train =os.path.join(path,"train")
try:
    os.mkdir(path_train)
except:
    print("this directory is already exist")
path_train_kid =os.path.join(path_train,"kid")
try:
    os.mkdir(path_train_kid)
except:
    print("this directory is already exist")
path_train_adult =os.path.join(path_train,"adult")
try:
    os.mkdir(path_train_adult)
except:
    print("this directory is already exist")
path_test = os.path.join(path,"test")
try:
    os.mkdir(path_test)
except:
    print("this directory is already exist")
path_test_kid = os.path.join(path_test,"kid")
try: 
    os.mkdir(path_test_kid)
except:
    print("this directory is already exist")
path_test_adult =os.path.join(path_test,"adult")
try:
    os.mkdir(path_test_adult)
except:
    print("this directory is already exist")


files.transferImages("E:/Dataset/kid","E:/"+folder_name+"/train/kid","E:/"+folder_name+"/test/kid")
files.transferImages("E:/Dataset/adult","E:/"+folder_name+"/train/adult","E:/"+folder_name+"/test/adult")
images, labels = to_matrix("E:/"+folder_name+"/train") # רשימה עם מטריצות תמונות תלת מימדיות
images= np.array(images)
print(images.shape)
labels= np.array(labels)
#labels.reshape(len(labels), 1)
print(labels)
print(labels.shape)


# net:
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(128, 128, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

print("if you want to train the model type 1 and if you want to test the model type 2")
answer=input()
while answer is not "1" and answer is not "2" :
    print("if you want to train the model type 1 and if you want to only test the model type 2")
    answer=input()
if answer is "1":    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(images, labels, epochs= 6, batch_size= 64)
    model.save_weights("weights")
    plt.plot(history.history['accuracy'])
    plt.title('model accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('accuracy')
    plt.show()
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()
    
      
else:    
    model.load_weights("weights")
    

test2 , labels2 = to_matrix("E:/"+folder_name+"/test")
res = model.predict(np.array(test2))
print("accuracy test= " +str(test(res,labels2)))


files.transferimagesBackToOG("E:/"+folder_name+"/train/kid","E:/Dataset/kid")
files.transferimagesBackToOG("E:/"+folder_name+"/test/kid","E:/Dataset/kid")
files.transferimagesBackToOG("E:/"+folder_name+"/train/adult","E:/Dataset/adult")
files.transferimagesBackToOG("E:/"+folder_name+"/test/adult","E:/Dataset/adult")







