#from config import cfg
from keras import layers, models
from capsule_layers import CapsuleLayer, PrimaryCap, Length#, Mask
from keras.regularizers import l2
import random



def CapsNet(input_shape, n_class, num_routing, vocab_size, embed_dim, max_len):

    x = layers.Input(shape = input_shape)

    embed = layers.Embedding(vocab_size, embed_dim, input_length = max_len)(x)

    # Layer 1: Conv1D layer
    conv = layers.Conv1D(filters = 64, kernel_size = 2, strides = 1, padding = 'same', activation = 'relu', name = 'conv1')(embed)

    # Layer 2: Dropout regularization
    dropout = layers.Dropout(0.5)(conv)

    # Layer 3: Primary layer with `squash` activation, then reshape to [None, num_capsule, dim_vector]
    primary_caps = PrimaryCap(dropout, dim_vector = 8, n_channels = 32, kernel_size = 9, strides = 2, padding = 'same', name = "primary_caps")

    # Layer 4: Capsule layer. Routing algorithm works here.
    category_caps = CapsuleLayer(num_capsule = n_class, dim_vector = 16, num_routing = num_routing, name = 'category_caps')(primary_caps)

    # Layer 5: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    out_caps = Length(name = 'out_caps')(category_caps)   #calcola la probabilità di ciascuna classe

    return models.Model(input = x, output = out_caps)