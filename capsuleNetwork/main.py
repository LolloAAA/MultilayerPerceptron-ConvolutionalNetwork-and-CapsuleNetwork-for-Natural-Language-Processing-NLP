import os
import math
from termcolor import colored
import numpy as np
import matplotlib
matplotlib.use('Agg')
from math import log2
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from keras import callbacks
from keras.utils.vis_utils import plot_model
from matplotlib import pyplot as plt
from keras import backend as K
from capsule_net import CapsNet
from keras.preprocessing.text import Tokenizer, one_hot
from keras.preprocessing.sequence import pad_sequences
import gensim
from sklearn.model_selection import train_test_split

#   Exponential Adaptive Learning Rate
def lr(epoch):
    return 0.001 * np.exp(-epoch / 10.)


def train(model, train, dev, test, optimizer, epoch, batch_size, schedule):
    (X_train, Y_train) = train
    (X_val, Y_val) = test
    (X_test, Y_test) = test

    lr_decay = callbacks.LearningRateScheduler(schedule = schedule, verbose = 0)    # Setting the adaptive learning rate

    model.compile(optimizer = optimizer, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    model.summary()

    history = model.fit(x = X_train, y = Y_train, validation_data = [X_val, Y_val], batch_size = batch_size, epochs = epoch, callbacks = [lr_decay], shuffle = True, verbose = True)
    loss, accuracy = model.evaluate(X_test, Y_test, verbose = True, batch_size = batch_size)
    print("Testing Accuracy:  {:.4f}".format(accuracy))




if __name__ == "__main__":
    #   Get dataset and save in "training"
    training = np.genfromtxt('../data_clean.csv', encoding = "utf8", delimiter = ';', skip_header = 0, usecols = (1,2), dtype = None, invalid_raise = False)
    
    X = np.asarray([str(x[0]) for x in training])   # Cleaned sentences
    y = np.asarray([str(x[1]) for x in training])   # Labels

    text_train, text_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, shuffle = True)

    etichette = text_test
    for i in range(len(etichette)):
        etichette[i] = text_test[i]

    tokenizer = Tokenizer(filters = '!"#$%&()*+,./:;<=>?@[\\]^`{|}~\t\n')
    tokenizer.fit_on_texts(text_train)
    x_train = tokenizer.texts_to_sequences(text_train)
    X_test = tokenizer.texts_to_sequences(text_test)

    word_index = tokenizer.word_index #Salvo il dizionario "word -> index"
    vocab_size = len(word_index) + 1  # Adding 1 because of reserved 0 index
    max_len = max(len(x) for x in x_train) # longest text in train set

    x_train = pad_sequences(x_train, padding = 'post', maxlen = max_len)
    x_test = pad_sequences(X_test, padding = 'post', maxlen = max_len)

    embedding_dim = 128

    #   input_shape = must be the dimension of every record, so it will be "max_len"
    #   n_class = is the number of labels for the classification. In this case is a binary problem, so the final classes are 2
    #   num_routing = number of iterations of routing-by-agreement. In the state-of-the-art, the value most used is 3 
    model = CapsNet(input_shape = x_train.shape[1:], n_class = 2, num_routing = 3, vocab_size = vocab_size, embed_dim = embedding_dim, max_len = max_len)
    train( model = model, train = (x_train, y_train), dev = (x_test, y_test), test = (x_test, y_test), optimizer = "adam", epoch = 50, batch_size = 64, schedule = lr)
