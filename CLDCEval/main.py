"""
Multi-purpose Code for the evaluation of Multilingual Models on Cross Language Document Classification:

    - Executes Data Preparation and PreProcessing
    - Two training and evaluation modes:
        - Monolingual Models on Language Independent Document Classification using MONOLINGUAL EMBEDDINGS
        - Train and Evaluates on one language at a time using MULTILINGUAL EMBEDDINGS:
            * Train on En -> Test on all
            * Train on En+De -> Test on all
            * Train on all -> Test on all
        - Comparing between models like:
            *  MLP Classifier
            *  Multi-Filter CNN over sentences + MLP Classifier
            *  Logistic Regression
            *  SVM

Created on Mon Feb 26 2018

@author: Meryem M'hamdi (Meryem.Mhamdi@epfl.ch)

"""

import cPickle as pkl
import os

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from keras.preprocessing.sequence import pad_sequences
from sklearn.svm import LinearSVC
from keras.utils import to_categorical

from Models.multi_filter_cnn_model import *
from Models.mlp_model import *
from dataModule import data_utils
from dataModule.RCV import rcv_processor
from get_args import *
from metrics import *
import numpy as np


def train_cldc_model(x_train_arr_dict, y_train_dict, x_dev_arr_dict, y_dev_dict,
                     x_test_arr_dict, y_test_dict, sequence_length):
    print("Preparing Train/ Dev/ Test")

    if len(train_lang) == 1:  ## Training and Validation on English and Testing on other languages
        print("Training and Validation on %s " % args.multi_train)
        x_train = x_train_arr_dict[train_lang[0]]
        y_train = y_train_dict[train_lang[0]]
        x_dev = x_dev_arr_dict[train_lang[0]]
        y_dev = y_dev_dict[train_lang[0]]

    else:  ## Training and validation on at least two languages and Testing on all other languages
        print("Training and Validation on %s " % args.multi_train)

        x_train = np.concatenate([x_train_arr_dict[train_lang[i]] for i in range(0, len(train_lang))], axis=0)
        y_train = np.concatenate([y_train_dict[train_lang[i]] for i in range(0, len(train_lang))], axis=0)
        x_dev = np.concatenate([x_dev_arr_dict[train_lang[i]] for i in range(0, len(train_lang))], axis=0)
        y_dev = np.concatenate([y_dev_dict[train_lang[i]] for i in range(0, len(train_lang))], axis=0)

    n_classes = len(set(list(y_train)))
    print("n_classes=", n_classes)

    one_hot_train = to_categorical(list(y_train), num_classes=n_classes)
    one_hot_dev = to_categorical(list(y_dev), num_classes=n_classes)

    y_test_one_hot_dict = {}
    for lang in test_lang_list:
        one_hot_test = to_categorical(list(y_test_dict[lang]), num_classes=n_classes)
        y_test_one_hot_dict.update({lang: one_hot_test})

    if args.model_choice == "cnn":
        x_train_exp = np.expand_dims(x_train, axis=3)
        x_dev_exp = np.expand_dims(x_dev, axis=3)

        print('x_train_exp.shape=', x_train_exp.shape)
        print('x_dev_exp.shape=', x_dev_exp.shape)

        # Testing on different languages
        x_test_exp_dict = {}
        for lang in test_lang_list:
            x_test_exp_dict.update({lang: np.expand_dims(x_test_arr_dict[lang], axis=3)})
            cldc_model = KerasMultiFilterCNNModel(args, sequence_length, n_classes)

    else:  ### mlp
        cldc_model = MLPModel(args, MAX_SEQUENCE_LENGTH, n_classes, word_index, embedding_matrix)
        x_train_exp = x_train
        x_dev_exp = x_dev

        x_test_exp_dict = {}
        for lang in test_lang_list:
            x_test_exp_dict.update({lang: x_test_arr_dict[lang]})

    ## Calling Metrics class
    metrics = Metrics(x_train_exp, one_hot_train, x_dev_exp, one_hot_dev, x_test_exp_dict,
                      y_test_one_hot_dict, args.mode, n_classes)

    # print(cldc_model.model.summary())
    ## Fitting the model to train
    print("Fitting the model ....")

    history = cldc_model.model.fit(x_train_exp, one_hot_train,
                                   batch_size=args.batch_size,
                                   epochs=args.epochs,
                                   shuffle=True,
                                   validation_data=(x_dev_exp, one_hot_dev),
                                   callbacks=[metrics])

    # print(history.history)

    # Evaluate the model on training dataset
    scores_train = cldc_model.model.evaluate(x_train_exp, one_hot_train, verbose=0)
    print("%s: %.2f%%" % (cldc_model.model.metrics_names[1], scores_train[1] * 100))

    # Evaluate the model on validation dataset
    scores_val = cldc_model.model.evaluate(x_dev_exp, one_hot_dev, verbose=0)
    print("%s: %.2f%%" % (cldc_model.model.metrics_names[1], scores_val[1] * 100))

    # Evaluate the model on all languages
    for lang in test_lang_list:
        scores_test = cldc_model.model.evaluate(x_test_exp_dict[lang], y_test_one_hot_dict[lang], verbose=0)
        print("Evaluating for %s with %s: %.2f%%" % (lang, cldc_model.model.metrics_names[1], scores_test[1] * 100))

    # serialize model to YAML
    model_yaml = cldc_model.model.to_yaml()
    with open(args.model_save_path + args.model_choice.upper() + "_Keras_Models_RCV/" + args.multi_train + "_" +
                      args.multi_model_file + "_" + args.model_file, "w") as yaml_file:
        yaml_file.write(model_yaml)

    return metrics, cldc_model, history


if __name__ == '__main__':

    """ 1. Processing command line arguments, extracting Training Languages and ISO languages name dictionary """
    global args, lang_list, lang_dict, model_lang_mapping, train_lang, test_lang_list

    args = get_args()

    lang_dict = {}
    with open("../../iso_lang_abbr.txt") as iso_lang:
        for line in iso_lang:
            lang_dict.update({line.split(":")[1][:-1]: line.split(":")[0]})

    model_results_dir = args.model_save_path + args.model_choice.upper() + "_Keras_Models_RCV/"
    if not os.path.isdir(model_results_dir):
        os.makedirs(model_results_dir)

    """ 2. Checking Train mode """
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

    """ 3. Data Extraction """
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

        dp = rcv_processor.RCVProcessor(data_util, args.mode)

        x_train_dict.update({lang_dict[language]: dp.x_vec_train})
        y_train_dict.update({lang_dict[language]: dp.label_ids_train})
        x_dev_dict.update({lang_dict[language]: dp.x_vec_dev})
        y_dev_dict.update({lang_dict[language]: dp.label_ids_dev})
        x_test_dict.update({lang_dict[language]: dp.x_vec_test})
        y_test_dict.update({lang_dict[language]: dp.label_ids_test})
        data_util_dict.update({lang_dict[language]: data_util})

    # Fixed Size input Matrices for train, test and dev
    # Compute mean number of sentences
    
    test_lang_list_new = []
    for lang in test_lang_list:
        test_lang_list_new.append(lang_dict[lang])
    
    test_lang_list = test_lang_list_new

    x_comb = []
    for lang in test_lang_list:
        x_comb += x_train_dict[lang] + x_dev_dict[lang] + x_test_dict[lang]

    max_sentences_list = []
    for doc in x_comb:  ## document level
        max_sentences_list.append(len(doc))
        # if len(doc)> max_sentences:
        #     max_sentences = len(doc)

    max_sentences = int(np.mean(max_sentences_list))
    print("max_sentences =", max_sentences)


    """ 3. Vocabulary Creation """
    if args.mode == "mono":
        vocab_path = data_util.data_root+ args.language + "_word_vocab.p"
    else:
        vocab_path = data_util.data_root+ "multi_lang_word_vocab.p"

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

    x_train_arr_dict = {}
    x_dev_arr_dict = {}
    x_test_arr_dict = {}
    """
    for lang in test_lang_list:
        if args.model_choice == "cnn":
            ## Padding all sentences from all languages to create fixed size matrices
            print("Padding for different languages >>>>")
            x_train_arr = data_util_dict[lang].pad_to_fixed_size(x_train_dict[lang], max_sentences)
            x_dev_arr = data_util_dict[lang].pad_to_fixed_size(x_dev_dict[lang], max_sentences)
            x_test_arr = data_util_dict[lang].pad_to_fixed_size(x_test_dict[lang], max_sentences)
        else:
            ## No Padding is needed but taking the average over words
            print("No padding required. Only averaging over the list of words per each document")
            x_train_arr = np.array(data_util_dict[lang].sent_avg(x_train_dict[lang]))#x_train_dict[lang])
            x_dev_arr = np.array(data_util_dict[lang].sent_avg(x_dev_dict[lang]))# x_dev_dict[lang])
            x_test_arr = np.array(data_util_dict[lang].sent_avg(x_test_dict[lang]))# x_test_dict[lang])

        print("x_train_arr.shape= ", x_train_arr.shape)
        print("x_dev_arr.shape= ", x_dev_arr.shape)
        print("x_test_arr.shape= ", x_test_arr.shape)
        x_train_arr_dict.update({lang: x_train_arr})
        x_dev_arr_dict.update({lang: x_dev_arr})
        x_test_arr_dict.update({lang: x_test_arr})
    """

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
                if token in vocab:
                    list_ids_sub.append(vocab[token])
                else:
                    list_ids_sub.append(len(vocab)+1)
            sequences_train.append(list_ids_sub)

        print("Dev...")
        sequences_dev = []
        for doc in x_dev_dict[lang]:
            list_ids_sub = []
            if token in vocab:
                list_ids_sub.append(vocab[token])
            else:
                list_ids_sub.append(len(vocab)+1)
            sequences_dev.append(list_ids_sub)

        print("Test...")
        sequences_test = []
        for doc in x_test_dict[lang]:
            list_ids_sub = []
            if token in vocab:
                list_ids_sub.append(vocab[token])
            else:
                list_ids_sub.append(len(vocab)+1)
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
    # Train model that complies to the chosen language

    print ("Training Model... ")

    if args.model_choice == "svm":
        clf = LinearSVC(random_state=0)
        if len(train_lang) == 1:  ## Training and Validation on English and Testing on other languages
            print("Training and Validation on %s " % args.multi_train)
            x_train = x_train_arr_dict[train_lang[0]]
            y_train = y_train_dict[train_lang[0]]

        else:  ## Training and validation on at least two languages and Testing on all other languages
            print("Training and Validation on %s " % args.multi_train)

            x_train = np.concatenate([x_train_arr_dict[train_lang[i]] for i in range(0, len(train_lang))], axis=0)
            y_train = np.concatenate([y_train_dict[train_lang[i]] for i in range(0, len(train_lang))], axis=0)

        ## Fitting Model
        clf.fit(x_train, y_train)

        ## Predicting
        print("Predicting in x_train")
        y_train_pred = clf.predict(x_train)

        results_dict = {}
        results_dict['y_train_pred'] = y_train_pred
        results_dict['y_train_trg'] = y_train
        _acc = accuracy_score(y_train, y_train_pred)
        _f1_M, _f1_m = f1_score(y_train, y_train_pred, average='macro'), \
                       f1_score(y_train, y_train_pred, average='micro')
        _recall_M, _recall_m = recall_score(y_train, y_train_pred, average='macro'),\
                               recall_score(y_train, y_train_pred,average='micro')
        _precision_M, _precision_m = precision_score(y_train, y_train_pred, average='macro'), \
                                     precision_score(y_train, y_train_pred,  average='micro')

        results_dict['train_metrics'] = {"acc": _acc, "f1_macro": _f1_M, "f1_micro": _f1_m,
                                         "recall_macro": _recall_M, "recall_micro": _recall_m,
                                         "precision_macro": _precision_M, "precision_micro": _precision_m}
        for lang in test_lang_list:
            y_test_pred = clf.predict(x_test_arr_dict[lang])
            results_dict['y_test_pred_'+lang] = y_test_pred
            results_dict['y_test_trg_'+lang] = y_test_dict[lang]
            _acc = accuracy_score(y_test_dict[lang], y_test_pred)
            _f1_M, _f1_m = f1_score(y_test_dict[lang], y_test_pred, average='macro'), \
                           f1_score(y_test_dict[lang], y_test_pred, average='micro')
            _recall_M, _recall_m = recall_score(y_test_dict[lang], y_test_pred, average='macro'), \
                                   recall_score(y_test_dict[lang], y_test_pred, average='micro')
            _precision_M, _precision_m = precision_score(y_test_dict[lang], y_test_pred, average='macro'), \
                                         precision_score(y_test_dict[lang], y_test_pred,  average='micro')

            results_dict['test_metrics_'+lang] = {"acc": _acc, "f1_macro": _f1_M, "f1_micro": _f1_m,
                                             "recall_macro": _recall_M, "recall_micro": _recall_m,
                                             "precision_macro": _precision_M, "precision_micro": _precision_m}

    else:
        metrics, cldc_model, history = train_cldc_model(x_train_dict, y_train_dict, x_dev_dict, y_dev_dict,
                                                        x_test_dict, y_test_dict, max_sentences)

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
            results_dict['y_test_pred_' + lang] = metrics.test_preds_dict[lang]
            results_dict['y_test_trg_' + lang] = metrics.test_trgs_dict[lang]
            results_dict['test_metrics_' + lang] = metrics.test_metrics_dict[lang]

        # Saving losses
        results_dict['train_loss'] = history.history['loss']
        # results_dict['val_loss'] = history.history['val_loss']

        # save model
        cldc_model.model.save(save_path + "_" + args.model_weights_path)

    with open(save_path + "_results.p", "wb") as dict_pkl:
        pkl.dump(results_dict, dict_pkl)

    print("Saved model to disk")
