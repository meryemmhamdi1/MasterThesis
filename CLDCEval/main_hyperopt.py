from __future__ import print_function
import os

from Models.mlp_fine_tune_model import *
from Models.mlp_model import *
from Models.multi_filter_cnn_model import *
from Models.bi_gru_att_model import *
import vocab_embedding
from dataModule import data_utils
from dataModule.RCV import new_processor
from get_args import *
from metrics import *

from hyperopt import Trials, STATUS_OK, tpe
from keras.datasets import mnist
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from keras.utils import np_utils

from hyperas import optim
from hyperas.distributions import choice, uniform, conditional
import numpy as np

from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Embedding, Input, Dense, AveragePooling1D, Dropout, Flatten, Conv1D, MaxPooling1D, merge,\
    GRU, Bidirectional, TimeDistributed, Multiply, RepeatVector, Permute, Lambda
from hyperas.distributions import choice, uniform, conditional
import keras.backend as K


#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#os.environ["THEANO_FLAGS"] = "/device:GPU:1"

def get_data():
    global args
    args = get_args()

    lang_dict = {}

    with open("../iso_lang_abbr.txt") as iso_lang:
        for line in iso_lang:
            lang_dict.update({line.split(":")[1][:-1]: line.split(":")[0]})

    """ 2. Data Extraction """
    if args.mode == "mono":
        model_dir = args.w2v_dir
        model_lang_mapping = {'english': args.w2v_en, 'german': args.w2v_de, 'french': args.w2v_fr,
                              'italian': args.w2v_it}
        model_file = model_lang_mapping[args.language]
        train_lang = [lang_dict[args.language]]
        test_lang_list = [args.language]
    else:
        model_dir = args.model_dir
        model_file = args.multi_model_file
        train_lang = args.multi_train.split(",")
        test_lang_list = args.languages.split(',')

    """ 3. Embedding Loading """
    global embed_dim
    model, embed_dim = vocab_embedding.load_embeddings(args.mode, args.language, model_dir, model_file, lang_dict)
    """ I. Preprocessing """
    x_train_dict = {}
    y_train_dict = {}
    x_dev_dict = {}
    y_dev_dict = {}
    x_test_dict = {}
    y_test_dict = {}
    data_util_dict = {}

    if args.data_choice == "rcv":
        data_dir = args.data_rcv
    elif args.data_choice == "rcv-bal":
        data_dir = args.data_rcv_bal
    elif args.data_choice == "ted":
        data_dir = args.data_ted
    elif args.data_choice == "churn":
        data_dir = args.data_churn
    else:
        data_dir = args.data_dw

    for language in test_lang_list:
        print("Processing language=> ", language)
        data_util = data_utils.DataUtils(data_dir, args.pre_dir, args.stop_pos_path, args.lemma_use,
                                         args.stop_use, language, model_dir, model_file, embed_dim)

        dp = new_processor.RCVProcessor(data_util, lang_dict)

        x_train_dict.update({lang_dict[language]: dp.x_train_pro})
        y_train_dict.update({lang_dict[language]: dp.y_train})
        x_dev_dict.update({lang_dict[language]: dp.x_dev_pro})
        y_dev_dict.update({lang_dict[language]: dp.y_dev})
        x_test_dict.update({lang_dict[language]: dp.x_test_pro})
        y_test_dict.update({lang_dict[language]: dp.y_test})
        data_util_dict.update({lang_dict[language]: data_util})
        global n_classes
        n_classes = dp.n_classes

    print("3. Creation Global Vocabulary ...")
    x_train_all = []
    x_dev_all = []
    x_test_all = []
    for lang in x_train_dict:
        x_train_all += x_train_dict[lang]
        x_dev_all += x_dev_dict[lang]
        x_test_all += x_test_dict[lang]

    x_all = x_train_all + x_dev_all + x_test_all
    global vocab
    global vocab_dict
    vocab, vocab_dict = vocab_embedding.create_vocabulary(x_all)

    global max_sequences
    max_sequences = max([len(doc) for doc in x_all])

    print("max_sequences=", max_sequences)

    print("4. Converting to ids and Padding to fixed length ...")
    sequences_train_dict = {}
    sequences_dev_dict = {}
    sequences_test_dict = {}
    for lang in x_train_dict:
        sequences_train, sequences_dev, sequences_test = \
            vocab_embedding.convert_ids(x_train_dict[lang], x_dev_dict[lang], x_test_dict[lang], vocab)

        data_train, data_dev, data_test = vocab_embedding.pad_fixed_length(sequences_train, sequences_dev,
                                                                           sequences_test, len(vocab), max_sequences)

        sequences_train_dict.update({lang: data_train})
        sequences_dev_dict.update({lang: data_dev})
        sequences_test_dict.update({lang: data_test})

    print("6. Building Embedding Matrix")
    global embedding_matrix
    embedding_matrix = vocab_embedding.build_embedding_matrix(vocab, model, embed_dim, vocab_dict)

    if len(train_lang) == 1:  ## Training and Validation on English and Testing on other languages
        print("Training and Validation on %s " % args.multi_train)
        x_train = sequences_train_dict[train_lang[0]]
        y_train = y_train_dict[train_lang[0]]
        x_dev = sequences_dev_dict[train_lang[0]]
        y_dev = y_dev_dict[train_lang[0]]

    else:  ## Training and validation on at least two languages and Testing on all other languages
        print("Training and Validation on %s " % args.multi_train)
        x_train = np.concatenate([sequences_train_dict[train_lang[i]] for i in range(0, len(train_lang))], axis=0)
        y_train = np.concatenate([y_train_dict[train_lang[i]] for i in range(0, len(train_lang))], axis=0)
        x_dev = np.concatenate([sequences_dev_dict[train_lang[i]] for i in range(0, len(train_lang))], axis=0)
        y_dev = np.concatenate([y_dev_dict[train_lang[i]] for i in range(0, len(train_lang))], axis=0)

    x_train = x_train.reshape(len(x_train), max_sequences)
    x_dev = x_dev.reshape(len(x_dev), max_sequences)
    x_train = x_train.astype('float32')
    x_dev = x_dev.astype('float32')

    return x_train, y_train, x_dev, y_dev, sequences_test_dict, y_test_dict


def create_mlp_model(x_train, y_train, x_dev, y_dev, sequences_test_dict, y_test_dict):
    sequence_input = Input(shape=(max_sequences,), dtype='int32')
    embedding_layer = Embedding(len(vocab)+1, embed_dim, input_length=max_sequences,
                                weights=[embedding_matrix], trainable=False, mask_zero=False)(sequence_input)

    average_emb = AveragePooling1D(pool_size=max_sequences)(embedding_layer)

    dense_layer = Dense({{choice([256, 512, 1024])}}, input_shape=(embed_dim,),
                        activation={{choice(["relu", "sigmoid", "tanh"])}})(average_emb)

    dropout_layer = Dropout({{uniform(0, 1)}})(dense_layer)

    flat = Flatten()(dropout_layer)

    softmax_layer = Dense(n_classes, activation='softmax')(flat)

    cldc_model = Model(inputs=sequence_input, outputs=softmax_layer)

    adam = Adam(lr=args.learning_rate, beta_1=args.beta_1, beta_2=args.beta_2, epsilon=args.epsilon)

    #cldc_model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=["accuracy"])

    cldc_model.compile(optimizer={{choice(['rmsprop', 'adam', 'sgd'])}}, loss='categorical_crossentropy',
                       metrics=["accuracy"])

    cldc_model.fit(x_train, y_train,
                   batch_size={{choice([32, 64, 128])}},
                   epochs=1,
                   shuffle=True,
                   verbose=2,
                   validation_data=(x_dev, y_dev))

    score, acc = cldc_model.evaluate(sequences_test_dict["de"], y_test_dict["de"], verbose=0)
    print('Validation accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': cldc_model}


def create_cnn_model(x_train, y_train, x_dev, y_dev, sequences_test_dict, y_test_dict):
    sequence_input = Input(shape=(max_sequences,), dtype='int32')
    embedding_layer = Embedding(len(vocab)+1, embed_dim, input_length=max_sequences,
                                weights=[embedding_matrix], trainable=False, mask_zero=False)(sequence_input)

    filter_sizes = args.filter_sizes.split(',')
    space_feature_maps = {{choice([50,100, 200, 300])}}
    filter_size_1 = {{choice(list(range(3, 20, 2)))}}
    filter_size_2 = {{choice(list(range(4, 20, 2)))}}
    filter_size_3 = {{choice(list(range(5, 20, 2)))}}
    activ_fun = {{choice(["relu", "sigmoid", "tanh"])}}
    conv_0 = Conv1D(space_feature_maps, filter_size_1, padding='valid', kernel_initializer='normal', activation=activ_fun)(embedding_layer)

    conv_1 = Conv1D(space_feature_maps, filter_size_2, padding='valid', kernel_initializer='normal', activation=activ_fun)\
        (embedding_layer)

    conv_2 = Conv1D(space_feature_maps, filter_size_3, padding='valid', kernel_initializer='normal', activation=activ_fun)\
        (embedding_layer)

    maxpool_0 = MaxPooling1D(pool_size=max_sequences - filter_size_1 + 1, strides=1, padding='valid')(conv_0)
    maxpool_1 = MaxPooling1D(pool_size=max_sequences - filter_size_2 + 1, strides=1, padding='valid')(conv_1)
    maxpool_2 = MaxPooling1D(pool_size=max_sequences - filter_size_3 + 1, strides=1, padding='valid')(conv_2)


    merged_tensor = merge([maxpool_0, maxpool_1, maxpool_2], mode='concat', concat_axis=1)


    flatten = Flatten()(merged_tensor)

    # average_pooling = AveragePooling2D(pool_size=(sequence_length,1),strides=(1,1),
    #                                    border_mode='valid', dim_ordering='tf')(inputs)
    #
    # reshape = Reshape()(average_pooling)
    #reshape = Reshape((3*num_filters,))(merged_tensor)
    dropout_layer = Dropout({{uniform(0, 1)}})(flatten)
    softmax_layer = Dense(output_dim=n_classes, activation='softmax')(dropout_layer)

    # this creates a model that includes
    model = Model(inputs=sequence_input, outputs=softmax_layer)
    adam = Adam(lr=args.learning_rate, beta_1=args.beta_1, beta_2=args.beta_2, epsilon=args.epsilon)

    model.compile(optimizer="sgd", loss='categorical_crossentropy', metrics=["accuracy"])

    model.fit(x_train, y_train,
                   batch_size={{choice([32, 64, 128])}},
                   epochs=1,
                   shuffle=True,
                   verbose=2,
                   validation_data=(x_dev, y_dev))

    score, acc = model.evaluate(sequences_test_dict["de"], y_test_dict["de"], verbose=0)
    print('Validation accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


def create_gru_att_model(x_train, y_train, x_dev, y_dev, sequences_test_dict, y_test_dict):
    sequence_input = Input(shape=(max_sequences,), dtype='int32')
    embedding_layer = Embedding(len(vocab)+1, embed_dim, input_length=max_sequences,
                                weights=[embedding_matrix], trainable=False, mask_zero=False)(sequence_input)

    num_units = {{choice(list(range(100, 1000, 50)))}}
    dropout_perc = {{uniform(0, 1)}}
    if args.bidirectional:
        print("Building B-GRU model...")
        gru_out = Bidirectional(
            GRU(num_units, return_sequences=True, dropout=dropout_perc,
                recurrent_dropout=dropout_perc))(
            embedding_layer)

        print("Building attention")

        att = TimeDistributed(Dense(2 * num_units, activation='tanh'))(gru_out)
        att = TimeDistributed(Dense(1, activation='linear'))(att)
        att = Flatten()(att)
        att = Activation('softmax')(att)
        att = RepeatVector(2 * num_units)(att)
        att = Permute([2, 1])(att)

        m = Multiply()([att, gru_out])
        m = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(2 * num_units,))(m)

    else:
        print("Building GRU model...")
        gru_out = GRU(num_units, return_sequences=True, dropout=dropout_perc,
                      recurrent_dropout=dropout_perc)(embedding_layer)

        print("Building attention")

        att = TimeDistributed(Dense(num_units, activation='tanh'))(gru_out)
        att = TimeDistributed(Dense(1, activation='linear'))(att)
        att = Flatten()(att)
        att = Activation('softmax')(att)
        att = RepeatVector(num_units)(att)
        att = Permute([2, 1])(att)

        m = Multiply()([att, gru_out])
        m = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(num_units,))(m)

    dropout_layer = Dropout(dropout_perc)(m)
    softmax_layer = Dense(output_dim=n_classes, activation='softmax')(dropout_layer)

    model = Model(inputs=sequence_input, outputs=softmax_layer)

    model.compile(optimizer={{choice(['rmsprop', 'adam', 'sgd'])}}, loss='categorical_crossentropy', metrics=["accuracy"])

    model.fit(x_train, y_train,
              batch_size={{choice([32, 64, 128])}},
              epochs=1,
              shuffle=True,
              verbose=2,
              validation_data=(x_dev, y_dev))

    score, acc = model.evaluate(sequences_test_dict["de"], y_test_dict["de"], verbose=0)
    print('Validation accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

if __name__ == '__main__':

    lang_dict = {}

    with open("../iso_lang_abbr.txt") as iso_lang:
        for line in iso_lang:
            lang_dict.update({line.split(":")[1][:-1]: line.split(":")[0]})

    best_run, best_model = optim.minimize(model=create_gru_att_model,
                                          data=get_data,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=Trials())
    print(best_run)
