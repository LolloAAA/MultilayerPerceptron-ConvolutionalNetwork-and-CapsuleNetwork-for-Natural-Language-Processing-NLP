
import json
import keras
import keras.preprocessing.text as kpt
from keras.preprocessing.text import Tokenizer
import numpy as np
import csv
import simplejson
from tempfile import TemporaryFile
import math
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import time
from gensim.models import KeyedVectors
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.utils.vis_utils import plot_model
from matplotlib import pyplot as plt
from keras import callbacks


###################################################################################################################################################################
########################################################## Multilayer Perceptron with 2 hidden layer ##############################################################
################################################################## 9 hand-crafted features ########################################################################
################################################################# Word2Vec model pre-created ######################################################################
###################################################################################################################################################################


#   Dataset contains cleaned sentences and 9 hand-crafted features. With "usecols" parameter, I choose which features in the .csv file use in this neural network
#   The explanation of every features is showed in the below code
training = np.genfromtxt('../data_clean_with_feat.csv',encoding = "utf8", delimiter = ',', skip_header = 0, usecols=(2, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11), dtype=None, invalid_raise = False)

#   Save all the hand-crafted features value in list
#   All the hand-crafted features are calculater by me with some scripts. You can calculate all the features you want and how you want. They are called "hand-crafted" because are choosen and calculated by the programmator
#   Sentence lenght
lunghezza = [str(x[2]) for x in training]

#   Percentuage of word in a single sentence written in CAPS-LOCK 
capslock = [str(x[3]) for x in training]

#   Count of words in a sentence
numfrasi = [str(x[4]) for x in training]

#   Count of ? and ! in a sentence
numinter = [str(x[5]) for x in training]

#   Count of punctuation in a sentence
numpunt = [str(x[6]) for x in training]

#   Polarity of a sentence (-1 for negative, 0 neutral, +1 positive sentence)
polarita = [str(x[7]) for x in training]

#   Percentuage of error in a sentence
percenterrori = [str(x[8]) for x in training]

#   Percentuage of bad word in a sentence, calculated by a big dataset of italian bad-words
percentparolacce = [str(x[9]) for x in training]

# Save all sentence in a list
train_x = [str(x[1]) for x in training]

train_x10 = []

#########################################################################################################
############### Use a Word2Vec model for calculate sum and average value of every sentence ##############
import gensim
cc = 0
model = gensim.models.Word2Vec.load("../word2vec2final.model")
modelvec = gensim.models.Word2Vec.load("../word2vec2final.model")
for idx,val in enumerate(train_x):
    vettoreMedia = []
    vettoreSomma = []
    for i in range(0,128):
        vettoreMedia.append(0)
        vettoreSomma.append(0)

    if train_x[idx] != "":
        for r in train_x[idx].split():           
            vector = model.wv[r] / len(train_x)
            vettoreMedia = vector + vettoreMedia
            vettoreSomma = model.wv[r] + vettoreSomma            
    else:
        cc += 1
    vettoreMedia = np.append(vettoreMedia, vettoreSomma, axis=0)
    train_x10.append(vettoreMedia)
train_x = np.array(train_x10)
#########################################################################################################
#########################################################################################################


train_y = np.asarray([int(x[0]) for x in training])
   
####################################################################################################################################
##################### Manual-Normalization of the value of every features. Is not a mathematical normalization #####################
numFeatures = 8
count = 0
array = []
for t in train_x:
    lst = list(t)
    featureList = []

    a = percentparolacce[count]
    a = float(a)
    lst.append(int(round(a*1000)))
    featureList.append(int(round(a*1000)))

    b = polarita[count]
    b = float(b)
    featureList.append(int(round(b*100)))
    lst.append(int(round(b*100)))
    
    c = numfrasi[count]
    c = float(c)
    featureList.append(int(round(c)))
    lst.append(int(round(c))) 
    
    e = numinter[count]
    e = float(e)
    featureList.append(int(round(e)))
    lst.append(int(round(e)))
    
    f = percenterrori[count]
    f = float(f)
    featureList.append(int(round(f)))
    lst.append(int(round(f)))
    
    g = numpunt[count]
    g = float(g)
    featureList.append(int(round(g)))
    lst.append(int(round(g)))
    
    h = capslock[count]
    h = float(h)
    featureList.append(int(round(h*10)))
    lst.append(int(round(h*10)))
    
    i = lunghezza[count]
    i = float(i)
    featureList.append(int(round(i/10)))
    lst.append(int(round(i/10)))

    t = np.asarray(lst, dtype=np.float32)
    array.append(t)
    count +=1 

array = np.asarray(array)
train_x = array
#############################################################################################################
#############################################################################################################

X = train_x
Y = train_y

text_train, text_test, y_train, y_test = train_test_split(X, Y, test_size = 0.1, shuffle = True)

#############################################################################################################
#   Every record is represented as:
#   1. Sum of Word2Vec value of every word in the sentence (128 dimension)
#   2. Average of Word2Vec value of every word in the sentence (128 dimension)
#   3. 8 additional hand-crafted features 
#   Every record is 128 + 128 + 8 = 264 dimensionality.
#   So the firs input layer of the network must have 264 input nodes
#############################################################################################################
model = Sequential()
model.add(Dense(256, input_shape=(256 + numFeatures,), activation="relu"))
model.add(Dropout(0.45))
model.add(Dense(128, activation="sigmoid"))
model.add(Dropout(0.45))
model.add(Dense(56, activation="relu"))
model.add(Dropout(0.45))
model.add(Dense(2, activation='softmax'))

model.compile(loss = 'sparse_categorical_crossentropy', optimizer = "Adagrad", metrics = ['accuracy'])

model.summary()

history = model.fit(text_train,  y_train, batch_size = 64, epochs = 50, verbose = 1, validation_data = (text_test, y_test), shuffle = True)

#   Evaluate the MLP in training-set
train_loss, train_accuracy = model.evaluate(text_train, y_train, verbose = True)
#   Evaluate the MLP in test-set
test_loss, test_accuracy = model.evaluate(text_test, y_test, verbose = True)

print("TEST ACCURACY = "+str(test_accuracy))
