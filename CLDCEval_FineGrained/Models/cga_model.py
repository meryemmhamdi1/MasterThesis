import keras.backend as K
from keras.models import Model
from keras.optimizers import SGD, Adam

from keras.layers import Conv1D, concatenate, TimeDistributed, Multiply, RepeatVector, Average
from keras.layers import GRU, Bidirectional, GlobalMaxPooling1D
from keras.layers import Input, Embedding, Dense, Dropout, Permute, Activation, Lambda
from keras.layers.core import Flatten

class BiGRUAttModel(object):

    def __init__(self, max_sequences, word_index, embed_dim, embedding_matrix, bidirectional, num_of_units,
                 dropout, learning_rate, beta_1, beta_2, epsilon, n_classes, filter_sizes, num_filters):

        main_input = Input(shape=(max_sequences,), dtype='int32', name='main_input')

        embedding_layer = Embedding(len(word_index)+1, embed_dim, input_length=max_sequences,
                                    weights=[embedding_matrix], trainable=False)(main_input)

        embeddings = Dropout(dropout)(embedding_layer)

        filter_sizes = [2,3]

        if len(filter_sizes) == 1:

            print("Building CNN model with single kernel size...")

            kernel_size = filter_sizes[0]
            conv1d = Conv1D(num_filters, kernel_size, padding='valid', activation='relu')(embeddings)

            if bidirectional:
                print("Building B-GRU model...")

                gru_out = Bidirectional(GRU(num_of_units, return_sequences=True, dropout=dropout,
                                            recurrent_dropout=dropout))(conv1d)

                print("Building attention")
                att = TimeDistributed(Dense(2 * num_of_units, activation='tanh'))(gru_out)
                att = TimeDistributed(Dense(1, activation='linear'))(att)
                att = Flatten()(att)
                att = Activation('softmax')(att)
                att = RepeatVector(2 * num_of_units)(att)
                att = Permute([2, 1])(att)

                m = Multiply()([att, gru_out])
                m = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(2 * num_of_units,))(m)

            else:
                print("Building GRU model...")
                gru_out = GRU(num_of_units, return_sequences=True, dropout=dropout,
                              recurrent_dropout=dropout)(conv1d)

                print("Building attention")
                att = TimeDistributed(Dense(num_of_units, activation='tanh'))(gru_out)
                att = TimeDistributed(Dense(1, activation='linear'))(att)
                att = Flatten()(att)
                att = Activation('softmax')(att)
                att = RepeatVector(num_of_units)(att)
                att = Permute([2, 1])(att)

                m = Multiply()([att, gru_out])
                m = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(num_of_units,))(m)

            cg_out = Dropout(dropout)(m)
            predictions = Dense(2, activation='softmax')(cg_out)

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

        model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

        return model