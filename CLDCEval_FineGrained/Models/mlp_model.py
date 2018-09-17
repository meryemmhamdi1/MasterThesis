#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This script defines functions for creating, training and optimizing MLP using Keras library
    Created on Tue Feb 27 2018

    @author: Meryem M'hamdi (Meryem.Mhamdi@epfl.ch)
"""
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Embedding, Input, Dense, AveragePooling1D, Dropout, average, Flatten
#from hyperas.distributions import choice, uniform, conditional


class MLPModel(object):

    def __init__(self, embed_dim, max_sequences, dense, dropout, learning_rate, beta_1, beta_2, epsilon,
                 n_classes, word_index, embedding_matrix, single_label):

        if single_label:
            act_fun = "softmax"
            loss_fun = "categorical_crossentropy"
        else:
            act_fun = "sigmoid"
            loss_fun = "binary_crossentropy"

        sequence_input = Input(shape=(max_sequences,), dtype='int32')
        embedding_layer = Embedding(len(word_index)+1, embed_dim, input_length=max_sequences,
                                    weights=[embedding_matrix], trainable=False, mask_zero=False)(sequence_input)

        average_emb = AveragePooling1D(pool_size=max_sequences)(embedding_layer)

        #dense_layer = Dense(({{choice([256, 512, 1024])}}), input_shape=(embed_dim,), activation={{choice(['relu', 'sigmoid'])}})(average_emb)  # 512, "relu"

        #dense_layer = Dense(({{choice([256, 512, 1024])}}), input_shape=(embed_dim,), activation="relu")(average_emb)
        dense_layer = Dense(50, input_shape=(embed_dim,), activation="relu")(average_emb)

        dropout_layer = Dropout(dropout)(dense_layer)

        flat = Flatten()(dropout_layer)

        softmax_layer = Dense(n_classes, activation=act_fun)(flat)

        model = Model(input=sequence_input, output=softmax_layer)

        adam = Adam(lr=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)
        #sgd = SGD(lr=learning_rate, decay=0, momentum=0.9, nesterov=True)

        model.compile(optimizer=adam, loss=loss_fun, metrics=["accuracy"])

        self.model = model
