#from config import cfg
from keras import layers, models
from capsule_layers import CapsuleLayer, PrimaryCap, Length#, Mask
from keras.regularizers import l2
import random

#   In this function We define de capsule-network architecture.
def CapsNet(input_shape, n_class, num_routing, vocab_size, embed_dim, max_len):
    x = layers.Input(shape = input_shape)

    embed = layers.Embedding(vocab_size, embed_dim, input_length = max_len)(x)

    # Layer 1: Conv1D layer. This layer calculate the initial features used later by the capsules
    conv = layers.Conv1D(filters = 64, kernel_size = 2, strides = 1, padding = 'same', activation = 'relu', name = 'conv1')(embed)

    # Layer 2: Dropout regularization
    dropout = layers.Dropout(0.5)(conv)

    # Layer 3: First capsule-layer with `squash` activation.
    primary_caps = PrimaryCap(dropout, dim_vector = 8, n_channels = 32, kernel_size = 9, strides = 2, padding = 'same', name = "primary_caps")

    # Layer 4: Last output-capsule-layer. The number of capsules is the number of classes of the classitication.
    category_caps = CapsuleLayer(num_capsule = n_class, dim_vector = 16, num_routing = num_routing, name = 'category_caps')(primary_caps)

    # Layer 5: This is an auxiliary layer to replace each capsule with its length for the classification
    out_caps = Length(name = 'out_caps')(category_caps)   # This computes the probability of each class

    return models.Model(input = x, output = out_caps)