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


ciclo = 0

def lambda1(epoch):
    return 0.001 * np.exp(-epoch / 10.)

def plotFunc(history, batch_size, ciclo):
    import matplotlib.pyplot as plt
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('risultati/plotAcc__'+str(ciclo)+'__'+str(batch_size)+'.png')
    plt.close()
    #plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('risultati/plotLoss__'+str(ciclo)+'__'+str(batch_size)+'.png')
    plt.close()


class TestCallback(tf.keras.callbacks.Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % 5 == 0:
            print("CALBACK DI EPOCA ---> "+str(epoch))
            x, y , batch_size= self.test_data
            loss, acc = self.model.evaluate(x, y, verbose=0)
            file = open("risultati/risultati__2output__binarycross__emb=keras__e=50___bz="+str(batch_size)+".csv","a")
            file.write(str(round(acc, 4)).replace(".", ",")+"\n")
            file.close()


def train(model, train, dev, test, save_directory, optimizer, epoch, batch_size, schedule, ciclo):
    (X_train, Y_train) = train
    (X_val, Y_val) = test
    (X_test, Y_test) = test

    lr_decay = callbacks.LearningRateScheduler(schedule = schedule, verbose = 0)

    model.compile(optimizer = optimizer, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    import time
    start_time = time.time()
    history = model.fit(x = X_train, y = Y_train, validation_data = [X_val, Y_val], batch_size = batch_size, epochs = epoch, callbacks = [lr_decay], shuffle = True, verbose = False)
    print("FINISHHHHHHHHHH %s seconds ---" % (time.time() - start_time))
    '''
    loss, accuracy = model.evaluate(X_test, Y_test, verbose = False, batch_size = batch_size)

    plotFunc(history, batch_size, ciclo)

    file = open("risultati/risultati__2output__binarycross__emb=keras__e=50___bz="+str(batch_size)+".csv","a")
    file.write("-----------------\n")
    file.close()
    ciclo += 1
    print("########################################################################################################################################################################")
    '''


if __name__ == "__main__":

    training = np.genfromtxt('../../../dataset/data_train.csv', encoding = "utf8", delimiter = ';', skip_header = 0, usecols = (1,2), dtype = None, invalid_raise = False)
    
    X = np.asarray([str(x[0]) for x in training])
    y = np.asarray([str(x[1]) for x in training])

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


    #testSet_X = pad_sequences(testSet_X, padding = "post", maxlen = max_len)

    embedding_dim = 128

    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))

    model = gensim.models.Word2Vec.load("../../../dataset/word2vec2final.model")

    for word, i in word_index.items():
        try:
            embedding_vector = model.wv[word]
        except KeyError:
            embedding_vector = [0] * embedding_dim
        embedding_matrix[i] = embedding_vector

    #   input_shape = combacia con la dimensione di rappresentazione di ogni frase, quindi con max_len scelta
    #   n_class = numero delle categorie da classificare
    #   num_routing = numero di interazione del routing by agreement.


    o = 'adam'
    #e = [3, 5, 10, 15, 20, 50]   #sia con embedding che senza
    e = [50]
    bz = [64]
    s = lambda1

    tot = len(e) * len(bz)
    cnt = 0
    ciclo = 0


    for i in range(len(bz)):
        for j in range(len(e)):
            cnt += 1
            print(str(cnt)+" di "+str(tot))
            
            for k in range(1):
                ciclo += 1

                model = CapsNet(input_shape = x_train.shape[1:], n_class = 2, num_routing = 3, vocab_size = vocab_size, embed_dim = embedding_dim, max_len = max_len)

                #model.summary()
                #plot_model(model, to_file = 'grafo/model.png', show_shapes = True)
                folder = ""

                train( model = model, train = (x_train, y_train), dev = (x_test, y_test), test = (x_test, y_test), save_directory = folder, optimizer = o, epoch = e[j], batch_size = bz[i], schedule = s, ciclo=ciclo)