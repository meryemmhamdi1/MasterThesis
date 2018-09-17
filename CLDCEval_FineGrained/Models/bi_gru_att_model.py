#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This script defines functions for creating, training and optimizing multi-filter CNN model using Keras library
    Created on Tue Feb 27 2018

    @author: Meryem M'hamdi (Meryem.Mhamdi@epfl.ch)
"""
import keras.backend as K
from keras.models import Model
from keras.optimizers import SGD, Adam

from keras.layers import Conv1D, concatenate, TimeDistributed, Multiply, RepeatVector, Average
from keras.layers import GRU, Bidirectional, GlobalMaxPooling1D
from keras.layers import Input, Embedding, Dense, Dropout, Permute, Activation, Lambda
from keras.layers.core import Flatten

class BiGRUAttModel(object):

    def __init__(self, max_sequences, word_index, embed_dim, embedding_matrix, bidirectional, num_of_units,
                 dropout, learning_rate, beta_1, beta_2, epsilon, n_classes, single_label):


        if single_label:
            act_fun = "softmax"
            loss_fun = "categorical_crossentropy"
        else:
            act_fun = "sigmoid"
            loss_fun = "binary_crossentropy"

        sequence_input = Input(shape=(max_sequences,), dtype='int32')
        embedding_layer = Embedding(len(word_index)+1, embed_dim, input_length=max_sequences,
                                    weights=[embedding_matrix], trainable=False, mask_zero=False)(sequence_input)

        if bidirectional:
            print("Building B-GRU model...")
            gru_out = Bidirectional(
                GRU(num_of_units, return_sequences=True, dropout=dropout, recurrent_dropout=dropout))(
                embedding_layer)

            print("Building attention")

            att = TimeDistributed(Dense(2 * num_of_units, activation='tanh'))(gru_out)
            att = TimeDistributed(Dense(1, activation='linear'))(att)
            att = Flatten()(att)
            att = Activation(act_fun)(att)
            att = RepeatVector(2 * num_of_units)(att)
            att = Permute([2, 1])(att)

            m = Multiply()([att, gru_out])
            m = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(2 * num_of_units,))(m)

        else:
            print("Building GRU model...")
            gru_out = GRU(num_of_units, return_sequences=True, dropout=dropout,
                          recurrent_dropout=dropout)(embedding_layer)

            print("Building attention")

            att = TimeDistributed(Dense(num_of_units, activation='tanh'))(gru_out)
            att = TimeDistributed(Dense(1, activation='linear'))(att)
            att = Flatten()(att)
            att = Activation(act_fun)(att)
            att = RepeatVector(num_of_units)(att)
            att = Permute([2, 1])(att)

            m = Multiply()([att, gru_out])
            m = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(num_of_units,))(m)

        dropout_layer = Dropout(dropout)(m)
        last_layer = Dense(output_dim=n_classes, activation=act_fun)(dropout_layer)

        model = Model(input=sequence_input, output=last_layer)
        adam = Adam(lr=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)

        model.compile(optimizer=adam, loss=loss_fun, metrics=["accuracy"])

        self.model = model

