#KERAS SEQUENTIALS
import cv2
import os
import numpy as np
import pandas as pd
import matpltlib.pyplot as plt
import tensorflow as tf

#CREATE AN ARRAY OF LABELS OF ALL IMAGES
image_name = os.listdir(r'D:\Data Analytics')
len(image_name)
label = []
for i in image_name:
    if i.split('_')[0] == 'cats':
        label.append(0)
    elif i.split('_')[0] == 'dogs':
        label.append(1)
    else:
        label.append(2)

#LABELS
Y = np.array(label)
np.array(label).shape
loc = r'D:\Data Analytics'

#FEATURES
features = []
for i in os.listdir(loc):
    x = os.path.join(loc,i)
    f = cv2.resize(f,(100,100))
    features.append(rf)
X = np.array(features)/255
np.array(features).shape
form sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size = 0.20)
model = tf.keras.model.Sequential()

#INPUT LAYER
model.add(tf.keras.layers.Flatten())

#HIDDEN LAYER
model.add(tf.keras.layers.Dense(128, activation = 'relu'))
model.add(tf.keras.layers.Dense(64, activation = 'relu'))

#OUTPUT LAYER
model.add(tf.keras.layers.Dense(3, activation = 'softmax'))
model.compile(loss = 'sparse_categorical_crossentropy',
             optimizer = 'adam',
             metrics = ['accuracy'])
model.fit(xtrain,ytrain,epochs = 5)
ypred = model.predict(xtest)
#np.argmax(ypred[0])
y_predicted = []
for i in ypred:
    y_predicted.append(np.argmax(i))
(y_predicted == ytest).sum()/len(test)
len(y_predicted)
k = y_predicted[6]
if k == 0:
    p = 'cats'
elif k == 1:
    p = 'dogs'
else:
    p = 'pandas'
print('Prediction of image : ',p)
plt.imshow(xtest[6])
plt.show()


#Decision Tree Classifer
import cv2
import os
import tensorflow as tf
import numpy as np

#CREATE AN ARRAY OF LABELS OF ALL THE IMAGES
image_name = os.listdir(r'D:\Data Analytics')
len(image_name)
label = []
for i in image_name:
    if i.split('_')[0] == 'cats':
        label.append(0)
    elif i.split('_')[0] == 'dogs':
        label.append(1)
    else:
        label.append(2)

#LABEL
Y = np.array(label)
loc = r'D:\Data Analytics'

#FEATURES
features = []
for i in os.listdir(loc):
    x = op.path.join(loc,i)
    f = cv2.imread(x,0)
    rf = cv2.resize(f,(100,100))
    features.append(rf)
X = np.array(features).reshape(3000,-1)
X.shape
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size = 0.20)

#DECISION TREE CLASSIFIER
from sklearn.tree import DecisionTreeClassifier
dmodel = DecisionTreeClassifier()
dmodel.fit(xtrain,ytrain)

#TRAINING ACCURACY
dmodel.score(xtrain,ytrain)

#TESTING ACCURACY
dmodel.score(xtest,ytest)

#RANDOM FOREST CLASSIFIER
from sklearn.esemble import RandomForestClassifier
rmodel = RandomForestClassifier()
rmodel.fit(xtrain,ytrain)

#TRAINING ACCURACY
rmodel.score(xtrain,ytrain)

#TESTING ACCURACY
rmodel.score(xtest,ytest)



#KNEIGHBORS CLASSIFIERS
from sklearn.neighbors import KNeighborsClassifier
kmodel = KNeighborsClassifier(n_neighbors = 3)
kmodel.fit(xtrain,ytrain)

#TRAINING ACCURACY
kmodel.score(xtrain,ytrain)

#TESTING ACCURACY
kmodel.score(xtest,ytest)