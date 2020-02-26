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

###################################################################################################################################################################
################################################# Convolutional Neural Network single channel with 3 convolutional layer ##########################################
################################################### kernel and bias regularizer + exponenzial adaptive learning rate ##############################################
###################################################################################################################################################################

#   In this CNN we'll use an Exponential Adaptive Learning Rate to get a more stable network 
def alr(epoch):
    return 0.001 * np.exp(-epoch / 10.)

#   We load all entire dataset
training = np.genfromtxt('data_clean.csv', encoding = "utf8", delimiter = ';', skip_header = 0, usecols = (1,2), dtype = None, invalid_raise = False)

#   In X we have all sentences, in Y the respective labels
X = np.asarray([str(x[0]) for x in training])
y = np.asarray([str(x[1]) for x in training])

#   Create training-set and test-set
text_train, text_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, shuffle = True)


tokenizer = Tokenizer(filters = '!"#$%&()*+,./:;<=>?@[\\]^`{|}~\t\n')
tokenizer.fit_on_texts(text_train)
X_train = tokenizer.texts_to_sequences(text_train)
X_test = tokenizer.texts_to_sequences(text_test)

word_index = tokenizer.word_index # Create the "word -> index" dictionary
vocab_size = len(word_index) + 1  # Adding 1 because of reserved 0 index
maxlen = max(len(x) for x in X_train) # Set the maxlen used for every sentences (in this case we compute this but we can decide it too)

#   Uniform every sentence (in train and test) to a fixed maxlen computed before
X_train = pad_sequences(X_train, padding = 'post', maxlen = maxlen)
X_test = pad_sequences(X_test, padding = "post", maxlen = maxlen)

#   In this project I used Word2Vec embedding pre-trained by myself. You can use a personal embedding
embedding_dim = 128
embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
model = gensim.models.Word2Vec.load("word2vec.model")

#   Set the word-weight computed by Word2Vec model in weights-matrix
for word, i in word_index.items():
    try:
        embedding_vector = model.wv[word]
    except KeyError:
        embedding_vector = [0] * embedding_dim
    embedding_matrix[i] = embedding_vector

#   Construct the Convolutional Neural Network
#   In Convolutional Layer we use kernel and bias regularizer for limit the overfitting
model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length = maxlen, weights = [embedding_matrix], trainable = False)) # This first layer create the sentence-matrix for convolutional operation
model.add(Conv1D(64, 2, activation = 'relu', strides = 1, padding = 'same', kernel_regularizer = l2(0.05), bias_regularizer = l2(0.05)))
model.add(MaxPooling1D())
model.add(Conv1D(512, 2, activation = 'relu', strides = 1, padding = 'same', kernel_regularizer = l2(0.05), bias_regularizer = l2(0.05)))
model.add(MaxPooling1D())
model.add(Conv1D(64, 2, activation = 'relu', strides = 1, padding = 'same', kernel_regularizer = l2(0.05), bias_regularizer = l2(0.05)))
model.add(MaxPooling1D())
                  
#   Classification with Fully Connected Network  
model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.45))
model.add(Dense(56, activation = "sigmoid"))
model.add(Dropout(0.45))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(optimizer = 'ADAM', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.summary()

#   Add the adaptive learning rate to the network
lr = callbacks.LearningRateScheduler(schedule = alr, verbose = 0)

#   Train the network
history = model.fit(X_train, y_train, epochs = 1, verbose = True, validation_data = (X_test, y_test), batch_size = 64, callbacks = [lr])

#   Evaluate the CNN in training-set
train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose = True)
#   Evaluate the CNN in test-set
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose = True)

print("TEST ACCURACY = "+str(test_accuracy))