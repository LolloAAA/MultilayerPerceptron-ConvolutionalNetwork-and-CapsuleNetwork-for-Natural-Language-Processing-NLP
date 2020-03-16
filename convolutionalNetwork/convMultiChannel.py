import pandas as pd
import numpy as np
import pickle
from keras import layers
from keras.models import Model
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer, one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, Flatten, MaxPooling1D, Input, Concatenate
from keras.models import load_model
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.externals import joblib
import os
from gensim.models import KeyedVectors
import gensim
os.environ['KMP_WARNINGS'] = 'off'
from keras.regularizers import l2
from keras import backend as K
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from keras import callbacks
from keras.utils.vis_utils import plot_model
from matplotlib import pyplot as plt

###################################################################################################################################################################
################################################# Convolutional Neural Network multi channel with 3 input channel #################################################
######################################################### 1. channel with stride-kernel = 2 #######################################################################
######################################################### 2. channel with stride-kernel = 3 #######################################################################
######################################################### 3. channel with sum of Word2Vec #########################################################################
################################################### kernel and bias regularizer + exponenzial adaptive learning rate ##############################################
###################################################################################################################################################################


###################################################################################################################################################################
################################################  For comments of every code-line, see "convSingleChannel.py" file  ###############################################
###################################################################################################################################################################

def lambda1(epoch):
    return 0.001 * np.exp(-epoch / 10.)

training = np.genfromtxt('../data_clean.csv', encoding = "utf8", delimiter = ';', skip_header = 0, usecols = (1,2), dtype = None, invalid_raise = False)

X = np.asarray([str(x[0]) for x in training])
y = np.asarray([str(x[1]) for x in training])

text_train, text_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, shuffle = True)

tokenizer = Tokenizer(filters = '!"#$%&()*+,./:;<=>?@[\\]^`{|}~\t\n')
tokenizer.fit_on_texts(text_train)
X_train = tokenizer.texts_to_sequences(text_train)
X_test = tokenizer.texts_to_sequences(text_test)

word_index = tokenizer.word_index
vocab_size = len(word_index) + 1
maxlen = max(len(x) for x in X_train)

X_train = pad_sequences(X_train, padding = 'post', maxlen = maxlen)

X_test = pad_sequences(X_test, padding = "post", maxlen = maxlen)

embedding_dim = 128

embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))

model = gensim.models.Word2Vec.load("../word2vec.model")

for word, i in word_index.items():
    try:
        embedding_vector = model.wv[word]
    except KeyError:
        embedding_vector = [0] * embedding_dim
    embedding_matrix[i] = embedding_vector

#   Word2Vec sum of train-senteces representation
X_trainSumEmb = []
for frase in X_train:
    tmp = [0] * embedding_dim
    for word in frase:
        if word != 0:
            s = list(word_index.keys())[list(word_index.values()).index(word)]
            try:
                tmp = np.sum([tmp, model.wv[s]], axis = 0)
            except KeyError:
                pass
    X_trainSumEmb.append(tmp)

#   Word2Vec sum of test-senteces representation
X_testSumEmb = []
for frase in X_test:
    tmp = [0] * embedding_dim
    for word in frase:
        if word != 0:
            s = list(word_index.keys())[list(word_index.values()).index(word)]
            try:
                tmp = np.sum([tmp, model.wv[s]], axis = 0)
            except KeyError:
                pass
    X_testSumEmb.append(tmp)

X_testSumEmb = np.asarray(X_testSumEmb)
X_trainSumEmb = np.asarray(X_trainSumEmb)



#   Input Channel 1 with kernel-stride = 2
inputs1 = Input(shape = (maxlen,))
embedding1 = Embedding(vocab_size, embedding_dim, input_length = maxlen, weights=[embedding_matrix], trainable = False)(inputs1)
conv1_1 = Conv1D(64, 2, activation = 'relu', strides = 1, padding = 'same', kernel_regularizer=l2(0.05), bias_regularizer=l2(0.05))(embedding1)
max1_1 = MaxPooling1D()(conv1_1)
conv1_2 = Conv1D(512, 2, activation = 'relu', strides = 1, padding = 'same', kernel_regularizer=l2(0.05), bias_regularizer=l2(0.05))(max1_1)
max1_2 = MaxPooling1D()(conv1_2)
conv1_3 = Conv1D(64, 2, activation = 'relu', strides = 1, padding = 'same', kernel_regularizer=l2(0.05), bias_regularizer=l2(0.05))(max1_2)
max1_3 = MaxPooling1D()(conv1_3)
flat1 = Flatten()(max1_3)

# Input Channel 2 with sentence Word2Vec sum
inputs2 = Input(shape = (embedding_dim,))

# Input Channel 3 with kernel-stride = 3
inputs3 = Input(shape = (maxlen,))
embedding2 = Embedding(vocab_size, embedding_dim, input_length = maxlen, weights=[embedding_matrix], trainable = False)(inputs3)
conv2 = Conv1D(64, 3, activation = 'relu', strides = 1, padding = 'same', kernel_regularizer=l2(0.05), bias_regularizer=l2(0.05))(embedding2)
max2 = MaxPooling1D()(conv2)
conv2 = Conv1D(512, 3, activation = 'relu', strides = 1, padding = 'same', kernel_regularizer=l2(0.05), bias_regularizer=l2(0.05))(max2)
max2 = MaxPooling1D()(conv2)
conv2 = Conv1D(64, 3, activation = 'relu', strides = 1, padding = 'same', kernel_regularizer=l2(0.05), bias_regularizer=l2(0.05))(max2)
max2 = MaxPooling1D()(conv2)
flat2 = Flatten()(max2)

#   Merge all 3 different input channel in only one to the Fully Connected for classification
merged = layers.concatenate([flat1, inputs2, flat2])
dense1 = Dense(256, activation = "relu")(merged)
drop1 = Dropout(0.45)(dense1)
dense2 = Dense(56, activation = "sigmoid")(drop1)
drop2 = Dropout(0.45)(dense2)
output = Dense(1, activation = 'sigmoid')(drop2)

model = Model(inputs = [inputs1, inputs2, inputs3], outputs = output)
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


model.summary()
lr = callbacks.LearningRateScheduler(schedule = lambda1, verbose = 0)
history = model.fit([X_train, X_trainSumEmb, X_train], y_train, epochs = 50, verbose = True, validation_data = ([X_test, X_testSumEmb, X_test], y_test), batch_size = 64, callbacks = [lr])

#   Evaluate the CNN in training-set
train_loss, train_accuracy = model.evaluate([X_train, X_trainSumEmb, X_train], y_train, verbose = True)
#   Evaluate the CNN in test-set
test_loss, test_accuracy = model.evaluate([X_test, X_testSumEmb, X_test], y_test, verbose = True)

print("TEST ACCURACY = "+str(test_accuracy))  
