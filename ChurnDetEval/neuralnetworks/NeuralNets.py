# -*- coding: utf-8 -*-
# from parameters import parameters

import keras.backend as K
from keras.layers import Conv1D, concatenate, TimeDistributed, Multiply, RepeatVector, Average
from keras.layers import GRU, Bidirectional, GlobalMaxPooling1D
from keras.layers import Input, Embedding, Dense, Dropout, Permute, Activation, Lambda
from keras.layers.core import Flatten
from keras.models import Model
from keras.optimizers import *
from keras.optimizers import Adam

from EarlyStoppingByPatience import \
    EarlyStoppingByPatience  # Chosse between F-score (arg. Task C) and accuracy for the rest
from TestCallback import TestCallback
from metrics_no_dev import *
from metrics import *


class NeuralNets(object):
    def __init__(self, filters, kernel_sizes, num_of_units, dropout, patience, embedding_matrix=None,
                 max_seq_length=None, max_num_words=None, word_embeddings_path=None,
                 batch_size=None, epochs=None, labels_to_ids=None, bidirectional=True):
        self.embedding_matrix = embedding_matrix
        self.max_seq_length = max_seq_length
        self.max_num_words = max_num_words
        self.word_embeddings_path = word_embeddings_path
        self.embedding_size = len(embedding_matrix[0])
        self.batch_size = batch_size
        self.epochs = epochs
        self.filters = filters
        self.kernel_sizes = kernel_sizes
        self.num_of_units = num_of_units
        self.dropout = dropout
        self.patience = patience
        self.bidirectional = bidirectional
        self.labels_to_ids = labels_to_ids
        self.id_to_label = {value: key for key, value in self.labels_to_ids.items()}
        self.num_of_classes = len(labels_to_ids)
        print(self.id_to_label)

    def train_model(self, network, x_train, y_train, x_dev, y_dev, x_test, y_test, mode, n_classes, lang_list):

        # Split data in train/validation
        # x_train, x_dev, y_train, y_dev = train_test_split(x, y, test_size=0.1, stratify=y, random_state=123)

        # Initailize and choose model
        if network == "cg_att":
            model = self.cg_att(x_train, y_train)
        elif network == "cnn":
            model = self.cnn(x_train, y_train)
        elif network == "gru_att":
            model = self.gru_att(x_train, y_train)
        else:
            model = self.cg(x_train, y_train)

        early_stopping = EarlyStoppingByPatience(x_dev, y_dev, self.patience, self.id_to_label, self.batch_size)
        #testing = TestCallback(x_test, y_test, self.id_to_label, self.batch_size)
        metrics = Metrics(x_train, y_train, x_dev, y_dev, x_test, y_test, mode, n_classes, self.id_to_label, self.batch_size)

        history = model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.epochs, verbose=1, shuffle=True,
                  callbacks=[metrics, early_stopping], validation_data=(x_dev, y_dev))

        # If fscore change to: "fscore = early_stopping.max_fscore"
        acc = early_stopping.max_acc

        results_dict = {}
        # Train Results
        results_dict['y_train_pred'] = metrics.train_preds
        results_dict['y_train_trg'] = metrics.train_trgs
        results_dict['train_metrics'] = metrics.train_metrics

        # Dev Results
        results_dict['y_dev_pred'] = metrics.val_preds
        results_dict['y_dev_trg'] = metrics.val_trgs
        results_dict['val_metrics'] = metrics.val_metrics

        # Test Results
        for lang in metrics.test_preds_dict:
            results_dict['y_test_pred_' + lang] = metrics.test_preds_dict[lang]
            results_dict['y_test_trg_' + lang] = metrics.test_trgs_dict[lang]
            results_dict['test_metrics_' + lang] = metrics.test_metrics_dict[lang]
            results_dict['test_bot_metrics_' + lang] = metrics.test_bot_metrics_dict[lang]
            results_dict['y_test_bot_pred_' + lang] = metrics.test_bot_preds_dict[lang]
            results_dict['y_test_bot_trg_' + lang] = metrics.test_bot_trgs_dict[lang]

        results_dict['train_loss'] = history.history['loss']

        return acc, results_dict, model

    def train_model_no_dev(self, network, x_train, y_train, x_test, y_test, x_bot_test, y_bot_test,
                           mode, n_classes, lang_list):

        # Split data in train/validation
        # x_train, x_dev, y_train, y_dev = train_test_split(x, y, test_size=0.1, stratify=y, random_state=123)

        # Initailize and choose model
        if network == "cg_att":
            model = self.cg_att(x_train, y_train)
        elif network == "cnn":
            model = self.cnn(x_train, y_train)
        elif network == "gru_att":
            model = self.gru_att(x_train, y_train)
        else:
            model = self.cg(x_train, y_train)

        early_stopping = EarlyStoppingByPatience(x_train, y_train, self.patience, self.id_to_label, self.batch_size)
        #testing = TestCallback(x_test, y_test, self.id_to_label, self.batch_size)
        metrics = MetricsNoDev(x_train, y_train, x_test, y_test, x_bot_test, y_bot_test, mode, n_classes)

        history = model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.epochs, verbose=1, shuffle=True,
                            callbacks=[metrics, early_stopping])

        # If fscore change to: "fscore = early_stopping.max_fscore"
        acc = early_stopping.max_acc

        results_dict = {}
        # Train Results
        results_dict['y_train_pred'] = metrics.train_preds
        results_dict['y_train_trg'] = metrics.train_trgs
        results_dict['train_metrics'] = metrics.train_metrics

        # Test Results
        for lang in metrics.test_preds_dict:
            results_dict['y_test_pred_' + lang] = metrics.test_preds_dict[lang]
            results_dict['y_test_trg_' + lang] = metrics.test_trgs_dict[lang]
            results_dict['test_metrics_' + lang] = metrics.test_metrics_dict[lang]
            results_dict['test_bot_metrics_' + lang] = metrics.test_bot_metrics_dict[lang]
            results_dict['y_test_bot_pred_' + lang] = metrics.test_bot_preds_dict[lang]
            results_dict['y_test_bot_trg_' + lang] = metrics.test_bot_trgs_dict[lang]

        results_dict['train_loss'] = history.history['loss']

        return acc, results_dict, model

    # CNN+GRU+attention
    def cg_att(self, x, y):

        main_input = Input(shape=(self.max_seq_length,), dtype='int32', name='main_input')
        embeddings = Embedding(self.max_num_words, self.embedding_size, input_length=self.max_seq_length,
                               weights=[self.embedding_matrix], trainable=False)(main_input)
        embeddings = Dropout(self.dropout)(embeddings)

        if len(self.kernel_sizes) == 1:

            print("Building CNN model with single kernel size...")

            kernel_size = self.kernel_sizes[0]
            conv1d = Conv1D(self.filters, kernel_size, padding='valid', activation='relu')(embeddings)

            if self.bidirectional:
                print("Building B-GRU model...")

                gru_out = Bidirectional(GRU(self.num_of_units, return_sequences=True, dropout=self.dropout,
                                            recurrent_dropout=self.dropout))(conv1d)

                print("Building attention")
                att = TimeDistributed(Dense(2 * self.num_of_units, activation='tanh'))(gru_out)
                att = TimeDistributed(Dense(1, activation='linear'))(att)
                att = Flatten()(att)
                att = Activation('softmax')(att)
                att = RepeatVector(2 * self.num_of_units)(att)
                att = Permute([2, 1])(att)

                m = Multiply()([att, gru_out])
                m = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(2 * self.num_of_units,))(m)

            else:
                print("Building GRU model...")
                gru_out = GRU(self.num_of_units, return_sequences=True, dropout=self.dropout,
                              recurrent_dropout=self.dropout)(conv1d)

                print("Building attention")
                att = TimeDistributed(Dense(self.num_of_units, activation='tanh'))(gru_out)
                att = TimeDistributed(Dense(1, activation='linear'))(att)
                att = Flatten()(att)
                att = Activation('softmax')(att)
                att = RepeatVector(self.num_of_units)(att)
                att = Permute([2, 1])(att)

                m = Multiply()([att, gru_out])
                m = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(self.num_of_units,))(m)

            cg_out = Dropout(self.dropout)(m)
            predictions = Dense(self.num_of_classes, activation='softmax')(cg_out)

        else:
            print("Building CNN model with multiple kernel sizes...")
            predictions = []
            convs = []
            for kernel_size in self.kernel_sizes:
                conv1d = Conv1D(self.filters, kernel_size, padding='valid', activation='relu')(embeddings)

                if self.bidirectional:
                    print("Building B-GRU model...")
                    gru_out = Bidirectional(GRU(self.num_of_units, return_sequences=True, dropout=self.dropout,
                                                recurrent_dropout=self.dropout))(conv1d)

                    print("Building attention")
                    att = TimeDistributed(Dense(2 * self.num_of_units, activation='tanh'))(gru_out)
                    att = TimeDistributed(Dense(1, activation='linear'))(att)
                    att = Flatten()(att)
                    att = Activation('softmax')(att)
                    att = RepeatVector(2 * self.num_of_units)(att)
                    att = Permute([2, 1])(att)

                    m = Multiply()([att, gru_out])
                    m = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(2 * self.num_of_units,))(m)

                    cg_out = m

                    predictions.append(Dense(self.num_of_classes, activation='softmax')(cg_out))

                else:
                    print("Building GRU model...")
                    gru_out = GRU(self.num_of_units, return_sequences=True, dropout=self.dropout,
                                  recurrent_dropout=self.dropout)(conv1d)

                    print("Building attention")
                    att = TimeDistributed(Dense(self.num_of_units, activation='tanh'))(gru_out)
                    att = TimeDistributed(Dense(1, activation='linear'))(att)
                    att = Flatten()(att)
                    att = Activation('softmax')(att)
                    att = RepeatVector(self.num_of_units)(att)
                    att = Permute([2, 1])(att)

                    m = Multiply()([att, gru_out])
                    m = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(self.num_of_units,))(m)

                    cg_out = m

                    predictions.append(Dense(self.num_of_classes, activation='softmax')(cg_out))

            predictions = Average()(predictions)

        print("Training GRU model...")

        model = Model(inputs=main_input, outputs=predictions)

        #model.summary(line_length=200)

        adam = Adam(lr=0.01, beta_1=0.7, beta_2=0.99, epsilon=1e-08)

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    def cnn(self, x, y):

        main_input = Input(shape=(self.max_seq_length,), dtype='int32', name='main_input')
        embeddings = Embedding(self.max_num_words, self.embedding_size, input_length=self.max_seq_length,
                               weights=[self.embedding_matrix], trainable=False)(main_input)

        if len(self.kernel_sizes) == 1:
            print("Building CNN model with single kernel size...")

            kernel_size = self.kernel_sizes[0]
            conv1d = Conv1D(self.filters, kernel_size, padding='valid', activation='relu')(embeddings)
            max_pool = GlobalMaxPooling1D()(conv1d)

        else:
            print("Building CNN model with multiple kernel sizes...")

            convs = []
            for kernel_size in self.kernel_sizes:
                conv1d = Conv1D(self.filters, kernel_size, padding='valid', activation='relu')(embeddings)
                max_pool = GlobalMaxPooling1D()(conv1d)
                convs.append(max_pool)

            max_pool = concatenate(convs)

        dropout_layer = Dropout(self.dropout)(max_pool)
        predictions = Dense(self.num_of_classes, activation='softmax')(dropout_layer)

        print("Training CNN model...")

        model = Model(inputs=main_input, outputs=predictions)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        #model.summary(line_length=200)

        return model

    # GRU + Attention
    def gru_att(self, x, y):

        main_input = Input(shape=(self.max_seq_length,), dtype='int32', name='main_input')
        embeddings = Embedding(self.max_num_words, self.embedding_size, input_length=self.max_seq_length,
                               weights=[self.embedding_matrix], trainable=True)(main_input)

        if self.bidirectional:
            print("Building B-GRU model...")
            gru_out = Bidirectional(
                GRU(self.num_of_units, return_sequences=True, dropout=self.dropout, recurrent_dropout=self.dropout))(
                embeddings)

            print("Building attention")

            att = TimeDistributed(Dense(2 * self.num_of_units, activation='tanh'))(gru_out)
            att = TimeDistributed(Dense(1, activation='linear'))(att)
            att = Flatten()(att)
            att = Activation('softmax')(att)
            att = RepeatVector(2 * self.num_of_units)(att)
            att = Permute([2, 1])(att)

            m = Multiply()([att, gru_out])
            m = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(2 * self.num_of_units,))(m)

        else:
            print("Building GRU model...")
            gru_out = GRU(self.num_of_units, return_sequences=True, dropout=self.dropout,
                          recurrent_dropout=self.dropout)(embeddings)

            print("Building attention")

            att = TimeDistributed(Dense(self.num_of_units, activation='tanh'))(gru_out)
            att = TimeDistributed(Dense(1, activation='linear'))(att)
            att = Flatten()(att)
            att = Activation('softmax')(att)
            att = RepeatVector(self.num_of_units)(att)
            att = Permute([2, 1])(att)

            m = Multiply()([att, gru_out])
            m = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(self.num_of_units,))(m)

        predictions = Dense(self.num_of_classes, activation='softmax')(m)

        print("Training GRU model...")

        model = Model(inputs=main_input, outputs=predictions)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        #model.summary(line_length=200)

        return model

    # CNN + GRU
    def cg(self, x, y):

        main_input = Input(shape=(self.max_seq_length,), dtype='int32', name='main_input')
        embeddings = Embedding(self.max_num_words, self.embedding_size, input_length=self.max_seq_length,
                               weights=[self.embedding_matrix], trainable=False)(main_input)
        embeddings = Dropout(self.dropout)(embeddings)

        if len(self.kernel_sizes) == 1:

            print("Building CNN model with single kernel size...")

            kernel_size = self.kernel_sizes[0]
            conv1d = Conv1D(self.filters, kernel_size, padding='valid', activation='relu')(embeddings)

            if self.bidirectional:
                print("Building B-GRU model...")

                gru_out = Bidirectional(GRU(self.num_of_units, return_sequences=True, dropout=self.dropout,
                                            recurrent_dropout=self.dropout))(conv1d)

                print("Building attention")

            else:
                print("Building GRU model...")
                gru_out = GRU(self.num_of_units, return_sequences=True, dropout=self.dropout,
                              recurrent_dropout=self.dropout)(conv1d)

                print("Building attention")

            cg_out = gru_out

        else:
            print("Building CNN model with multiple kernel sizes...")

            convs = []
            for kernel_size in self.kernel_sizes:
                conv1d = Conv1D(self.filters, kernel_size, padding='valid', activation='relu')(embeddings)

                if self.bidirectional:
                    print("Building B-GRU model...")
                    gru_out = Bidirectional(GRU(self.num_of_units, return_sequences=True, dropout=self.dropout,
                                                recurrent_dropout=self.dropout))(conv1d)

                    print("Building attention")

                else:
                    print("Building GRU model...")
                    gru_out = GRU(self.num_of_units, return_sequences=True, dropout=self.dropout,
                                  recurrent_dropout=self.dropout)(conv1d)

                    print("Building attention")

                convs.append(gru_out)

            cg_out = concatenate(convs)

        # cg_out = Dropout(dropout)(cg_out)
        predictions = Dense(self.num_of_classes, activation='softmax')(cg_out)

        print("Training GRU model...")

        model = Model(inputs=main_input, outputs=predictions)
        #model.summary(line_length=200)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        return model
