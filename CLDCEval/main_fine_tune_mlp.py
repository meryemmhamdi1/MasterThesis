"""
Multi-purpose Code for the evaluation of Multilingual Models on Cross Language Document Classification:

    - Executes Data Preparation and PreProcessing
    - Two training and evaluation modes:
        - Train and Evaluates on one language at a time using MULTILINGUAL EMBEDDINGS:
            * Train on En -> Test on all
            * Train on En+De -> Test on all
            * Train on all -> Test on all
        - Comparing between models like:
            * MLP with Fine Tuning

Created on Mon Feb 26 2018

@author: Meryem M'hamdi (Meryem.Mhamdi@epfl.ch)

"""
import cPickle as pkl

from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from tqdm import tqdm

from Models.mlp_fine_tune_model import *
from dataModule import data_utils
from dataModule.RCV import keras_preprocessor
from keras.callbacks import EarlyStopping
from keras.layers import Activation, Embedding, Input, Dense, AveragePooling1D, Lambda, Dropout
from get_args import *
from metrics import *
import os
from keras.models import Model
from gensim.models import KeyedVectors
import numpy as np

if __name__ == '__main__':

    """ 1. Processing command line arguments, extracting Training Languages and ISO languages name dictionary """

    global args, lang_list, lang_dict, model_lang_mapping, train_lang, test_lang_list

    args = get_args()

    lang_dict = {}
    with open("../iso_lang_abbr.txt") as iso_lang:
        for line in iso_lang:
            lang_dict.update({line.split(":")[1][:-1]: line.split(":")[0]})

    model_results_dir = args.model_save_path + args.model_choice.upper() + "_Keras_Models_RCV/"
    if not os.path.isdir(model_results_dir):
        os.makedirs(model_results_dir)

    """ 2. Data Extraction """
    if args.mode == "mono":
        model_dir = args.w2v_dir
        model_lang_mapping = {'english': args.w2v_en, 'german': args.w2v_de, 'french': args.w2v_fr, 'italian': args.w2v_it}
        model_file = model_lang_mapping[args.language]
        save_path = model_results_dir + args.language + "_mono"
        train_lang = [lang_dict[args.language]]
        test_lang_list = [args.language]
    else:
        model_dir = args.model_dir
        model_file = args.multi_model_file
        save_path = model_results_dir + args.multi_train + "_" + args.multi_model_file
        train_lang = args.multi_train.split(",")
        test_lang_list = args.languages.split(',')

    x_train_dict = {}
    y_train_dict = {}
    x_dev_dict = {}
    y_dev_dict = {}
    x_test_dict = {}
    y_test_dict = {}
    data_util_dict = {}

    for language in test_lang_list:
        print("Processing language=> ", language)
        data_util = data_utils.DataUtils(args.data_rcv, args.pre_dir, args.stop_pos_path, args.lemma_use,
                                         args.stop_use, language, model_dir, model_file, args.embed_dim)

        dp = keras_preprocessor.KerasProcessor(data_util, lang_dict)

        x_train_dict.update({lang_dict[language]: dp.x_train_doc})
        y_train_dict.update({lang_dict[language]: dp.label_ids_train})
        x_dev_dict.update({lang_dict[language]: dp.x_dev_doc})
        y_dev_dict.update({lang_dict[language]: dp.label_ids_dev})
        x_test_dict.update({lang_dict[language]: dp.x_test_doc})
        y_test_dict.update({lang_dict[language]: dp.label_ids_test})
        data_util_dict.update({lang_dict[language]: data_util})

    if args.mode == "mono":
        vocab_path = data_util.data_root+ args.language + "_word_vocab.p"
    else:
        vocab_path = data_util.data_root+ "multi_lang_word_vocab.p"

    """ 3. Vocabulary Creation """
    if not os.path.isfile(vocab_path):
        print("Building the vocabulary >>>>>>")
        X_train_all = []
        X_dev_all = []
        X_test_all = []
        for lang in x_train_dict.keys():
            X_train_all += x_train_dict[lang]
            X_dev_all += x_dev_dict[lang]
            X_test_all += x_test_dict[lang]

        x_all = X_train_all+X_dev_all+X_test_all

        vocab_dict = {}
        for doc in x_all:
            for token in doc:
                if token in vocab_dict:
                    vocab_dict[token] += 1
                else:
                    vocab_dict[token] = 1
        vocab_list = sorted(vocab_dict)
        vocab = dict([x, y] for (y, x) in enumerate(vocab_list))
        print("Saving the vocabulary")
        with open(vocab_path, "wb") as word_vocab_file:
            pkl.dump(vocab, word_vocab_file)
    else:
        print("Loading the vocabulary")
        with open(vocab_path, "rb") as word_vocab_file:
            vocab = pkl.load(word_vocab_file)
    word_index = vocab
    print("len(word_index)=", len(word_index))

    """ 4. Convertion to ids and padding"""
    if not os.path.isfile(data_util.data_root+"x_train_dict_pad.p") \
            or not os.path.isfile(data_util.data_root+"x_dev_dict_pad.p") \
            or not os.path.isfile(data_util.data_root+"x_test_dict_pad.p"):
        MAX_SEQUENCE_LENGTH = 0
        sequences_train_dict = {}
        sequences_dev_dict = {}
        sequences_test_dict = {}
        sequences_list = []
        lengths_list = []

        print("Convert the words to ids")
        for lang in x_train_dict.keys():
            print("Train...")
            sequences_train = []
            for doc in x_train_dict[lang]:
                list_ids_sub = []
                for token in doc:
                    list_ids_sub.append(vocab[token])
                sequences_train.append(list_ids_sub)

            print("Dev...")
            sequences_dev = []
            for doc in x_dev_dict[lang]:
                list_ids_sub = []
                for token in doc:
                    list_ids_sub.append(vocab[token])
                sequences_dev.append(list_ids_sub)

            print("Test...")
            sequences_test = []
            for doc in x_test_dict[lang]:
                list_ids_sub = []
                for token in doc:
                    list_ids_sub.append(vocab[token])
                sequences_test.append(list_ids_sub)

            sequences_train_dict.update({lang: sequences_train})
            sequences_dev_dict.update({lang: sequences_dev})
            sequences_test_dict.update({lang: sequences_test})

            sequences_list = sequences_train+sequences_dev+sequences_test
            lengths_list = [len(sequence) for sequence in sequences_list]
            max_lang = max(lengths_list)

            if max_lang > MAX_SEQUENCE_LENGTH:
                MAX_SEQUENCE_LENGTH = max_lang

        print("MAX_SEQUENCE_LENGTH= ", MAX_SEQUENCE_LENGTH)
        for lang in x_train_dict.keys():
            data_train = pad_sequences(sequences_train_dict[lang], padding='post', maxlen=MAX_SEQUENCE_LENGTH)
            data_dev = pad_sequences(sequences_dev_dict[lang], padding='post', maxlen=MAX_SEQUENCE_LENGTH)
            data_test = pad_sequences(sequences_test_dict[lang], padding='post', maxlen=MAX_SEQUENCE_LENGTH)

            x_train_dict.update({lang: data_train})
            x_dev_dict.update({lang: data_dev})
            x_test_dict.update({lang: data_test})

        print("Saving padded files>>>>")
        with open(data_util.data_root+"x_train_dict_pad.p", "wb") as x_train_pad_file:
            pkl.dump(x_train_dict, x_train_pad_file)

        with open(data_util.data_root+"x_dev_dict_pad.p", "wb") as x_dev_pad_file:
            pkl.dump(x_dev_dict, x_dev_pad_file)

        with open(data_util.data_root+"x_test_dict_pad.p", "wb") as x_test_pad_file:
            pkl.dump(x_test_dict, x_test_pad_file)

    else:
        print("Load files >>>>")
        with open(data_util.data_root+"x_train_dict_pad.p") as x_train_pad_file:
            x_train_dict = pkl.load(x_train_pad_file)

        with open(data_util.data_root+"x_dev_dict_pad.p") as x_dev_pad_file:
            x_dev_dict = pkl.load(x_dev_pad_file)

        with open(data_util.data_root+"x_test_dict_pad.p") as x_test_pad_file:
            x_test_dict = pkl.load(x_test_pad_file)

        MAX_SEQUENCE_LENGTH = len(x_train_dict[x_train_dict.keys()[0]][0])

    """ 5. Creation of Embedding Matrix"""
    model = {}
    if args.mode == "mono":
        if args.language == "english" or args.language == "german":
            model = KeyedVectors.load_word2vec_format(data_util.emb_model_path + data_util.emb_model_name,
                                                      binary=True)
        else:
            with open(data_util.emb_model_path + data_util.emb_model_name) as vector_file:
                word_vecs = vector_file.readlines()[1:]

            model = {}
            for word in word_vecs:
                parts = word.split(" ")
                # model.update({lang_dict[data_util.language]+"_"+parts[0]: map(float, parts[1:301])})
                model.update({parts[0]: map(float, parts[1:301])})
    else:
        with open(args.model_dir + args.multi_model_file) as model_file:
            data = model_file.readlines()

        print("Loading list of words and their vectors in all languages ....")
        for i in tqdm(range(0, len(data))):
            lang = data[i].split(" ")[0].split(":")[0]
            if lang in ["en", "fr", "de", "it"]:
                word = data[i].split(" ")[0].split(":")[1]
                vectors = [float(vector) for vector in data[i].split(" ")[1:]]
                model.update({word: vectors}) # model.update({lang+"_"+word: vectors})

    not_covered_words = []
    covered_words = []
    embedding_matrix = np.zeros((len(word_index) + 1, args.embed_dim))
    i = 0
    for word in word_index.keys():
        if word in model:
            embedding_vector = model[word]
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
            covered_words.append(word)
        else:
            not_covered_words.append(word)

        i = i + 1

    ## Save embedding coverage:
    with open(save_path + "_covered_words.p", "wb") as covered_file:
        pkl.dump(covered_words, covered_file)

    with open(save_path + "_not_covered_words.p", "wb") as not_covered_file:
        pkl.dump(not_covered_words, not_covered_file)

    if len(train_lang) == 1:  ## Training and Validation on English and Testing on other languages
        print("Training and Validation on %s " % args.multi_train)
        x_train = x_train_dict[train_lang[0]]
        y_train = y_train_dict[train_lang[0]]
        x_dev = x_dev_dict[train_lang[0]]
        y_dev = y_dev_dict[train_lang[0]]

    else:  ## Training and validation on at least two languages and Testing on all other languages
        print("Training and Validation on %s " % args.multi_train)
        x_train = np.concatenate([x_train_dict[train_lang[i]] for i in range(0, len(train_lang))], axis=0)
        y_train = np.concatenate([y_train_dict[train_lang[i]] for i in range(0, len(train_lang))], axis=0)
        x_dev = np.concatenate([x_dev_dict[train_lang[i]] for i in range(0, len(train_lang))], axis=0)
        y_dev = np.concatenate([y_dev_dict[train_lang[i]] for i in range(0, len(train_lang))], axis=0)

    n_classes = len(set(list(y_train)))
    one_hot_train = to_categorical(list(y_train), num_classes=n_classes)
    one_hot_dev = to_categorical(list(y_dev), num_classes=n_classes)
    y_test_one_hot_dict = {}
    for lang in test_lang_list:
        lang1 = lang_dict[lang]
        one_hot_test = to_categorical(list(y_test_dict[lang1]), num_classes=n_classes)
        y_test_one_hot_dict.update({lang1: one_hot_test})

    print("one_hot_train.shape=", one_hot_train.shape)
    ## MLP Model with trainable Embedding
    cldc_model = MLPFineTuneModel(args, MAX_SEQUENCE_LENGTH, n_classes, word_index, embedding_matrix)

    print("x_train.shape= ", x_train.shape)
    print("x_dev.shape= ", x_dev.shape)

    print("x_test_dict['en'].shape= ", x_test_dict['en'].shape)

    print(cldc_model.model.summary())

    metrics = Metrics(x_train, one_hot_train, x_dev, one_hot_dev, x_test_dict[lang_dict[args.language]],
                      y_test_one_hot_dict[lang_dict[args.language]], args.mode, n_classes)

    history = cldc_model.model.fit(x_train, one_hot_train,
                                   batch_size=args.batch_size,
                                   epochs=args.epochs,
                                   shuffle=True,
                                   validation_data=(x_dev, one_hot_dev),
                                   callbacks=[metrics])#, EarlyStopping(monitor='val_loss', patience=0)])

    # Evaluate the model on training dataset
    scores_train = cldc_model.model.evaluate(x_train, one_hot_train, verbose=0)
    print("%s: %.2f%%" % (cldc_model.model.metrics_names[1], scores_train[1] * 100))

    # Evaluate the model on validation dataset
    scores_val = cldc_model.model.evaluate(x_dev, one_hot_dev, verbose=0)
    print("%s: %.2f%%" % (cldc_model.model.metrics_names[1], scores_val[1] * 100))

    activations_dict = {}
    # Evaluate the model on all languages
    for lang in test_lang_list:
        lang1 = lang_dict[lang]
        scores_test = cldc_model.model.evaluate(x_test_dict[lang1], y_test_one_hot_dict[lang1], verbose=0)
        print("Evaluating for %s with %s: %.2f%%" % (lang1, cldc_model.model.metrics_names[1], scores_test[1] * 100))

        ### Saving the new document embeddings by building a new model with the activations of the old model
        ### this model is truncated after the first layer
        sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
        embedding_layer = Embedding(len(word_index)+1, args.embed_dim,
                                    weights=cldc_model.model.layers[1].get_weights(), input_length=MAX_SEQUENCE_LENGTH,
                                    trainable=True, mask_zero=True)(sequence_input)

        embed_zeroed = ZeroMaskedEntries()(embedding_layer)

        lambda_mean = Lambda(mask_aware_mean, mask_aware_mean_output_shape)(embed_zeroed)

        dense_layer = Dense(args.dense, input_shape=(args.embed_dim,), activation='relu',
                            weights=cldc_model.model.layers[4].get_weights())(lambda_mean)

        model2 = Model(input=sequence_input, output=dense_layer)

        activations_dict.update({lang1: model2.predict(x_test_dict[lang1])})

    # Get the fine tuned embeddings:
    fine_tuned_emb = cldc_model.model.layers[1].get_weights()[0]
    word_dict = {}
    i = 0
    for word in word_index.keys():
        word_dict.update({word: fine_tuned_emb[i]})
        i = i + 1

    with open(save_path + "_fine_tuned_emb.p", "wb") as fine_tuned_file:
        pkl.dump(word_dict, fine_tuned_file)

    # serialize model to YAML
    model_yaml = cldc_model.model.to_yaml()
    with open(args.model_save_path + args.model_choice.upper() + "_Keras_Models_RCV/" + args.multi_train + "_" +
              args.multi_model_file + "_" + args.model_file, "w") as yaml_file:
        yaml_file.write(model_yaml)


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
    for lang in test_lang_list:
        lang1 = lang_dict[lang]
        if args.mode == "multi":
            results_dict['y_test_pred_' + lang1] = metrics.test_preds_dict[lang1]
            results_dict['y_test_trg_' + lang1] = metrics.test_trgs_dict[lang1]
            results_dict['test_metrics_' + lang1] = metrics.test_metrics_dict[lang1]
        else:
            results_dict['y_test_pred_' + lang1] = metrics.test_preds
            results_dict['y_test_trg_' + lang1] = metrics.test_trgs
            results_dict['test_metrics_' + lang1] = metrics.test_metrics
        results_dict['activations_' + lang1] = activations_dict[lang1]


    # Saving losses
    results_dict['train_loss'] = history.history['loss']
    # results_dict['val_loss'] = history.history['val_loss']

    with open(save_path + "_results.p", "wb") as dict_pkl:
        pkl.dump(results_dict, dict_pkl)

    # serialize weights to HDF5
    cldc_model.model.save(save_path + "_" + args.model_weights_path)
