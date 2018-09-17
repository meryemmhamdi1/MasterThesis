import numpy as np
import random
import pandas as pd
import ast
from tqdm import tqdm
import collections
import string
from nltk.corpus import stopwords
import math
import nltk
import random as rnd
from string import punctuation

import numpy as np
from nltk.corpus import stopwords
from tqdm import tqdm
import csv
from keras.utils import to_categorical
import os
import cPickle as pkl


TRAIN_PERC = 0.6
DEV_PERC = 0.2
TEST_PERC = 0.2

lang_abb = {"en": "english", "fr": "french", "de": "german", "it": "italian"}
label_dict = {"CCAT": 0, "ECAT": 1, "GCAT": 2, "MCAT": 3}
lang_dict = {"english": "en", "french": "fr", "italian": "it", "german": "de"}

SEPARATOR = "\t"
MAX_SEQUENCES = 622


def num_there(s):
    return not any(i.isdigit() for i in s)


def all_punct(s):
    return not all(i in punctuation for i in s)


def read_cldc_docs(path, train_langs, test_langs):
    docs_train = {}
    y_train = {}
    docs_dev = {}
    y_dev = {}
    docs_test = {}
    y_test = {}
    if "," in train_langs:
        train_lang = train_langs.split(",")
    else:
        train_lang = [train_langs]
    if "," in test_langs:
        test_lang = test_langs.split(",")
    else:
        test_lang = [test_langs]
    for lang in train_lang:
        print("Loading train docs for language: ", lang)
        df = pd.read_csv(path + lang + "/train.csv")
        texts = list(df[df["lead_topic"] != "NONE"]["texts"])
        for text in texts:
            if lang not in docs_train:
                docs_train.update({lang: []})
            docs_train[lang].append(ast.literal_eval(text))
        y_train.update({lang: [label_dict[label] for label in list(df[df["lead_topic"] != "NONE"]["lead_topic"])]})

    for lang in train_lang:
        print("Loading dev docs for language: ", lang)
        df = pd.read_csv(path + lang + "/dev.csv")
        texts = list(df[df["lead_topic"] != "NONE"]["texts"])
        for text in texts:
            if lang not in docs_dev:
                docs_dev.update({lang: []})
            docs_dev[lang].append(ast.literal_eval(text))
        y_dev.update({lang: [label_dict[label] for label in list(df[df["lead_topic"] != "NONE"]["lead_topic"])]})

    for lang in test_lang:
        print("Loading test docs for language: ", lang)
        df = pd.read_csv(path + lang + "/test.csv")
        texts = list(df[df["lead_topic"] != "NONE"]["texts"])
        for text in texts:
            if lang not in docs_test:
                docs_test.update({lang: []})
            docs_test[lang].append(ast.literal_eval(text))
        y_test.update({lang: [label_dict[label] for label in list(df[df["lead_topic"] != "NONE"]["lead_topic"])]})

    return docs_train, y_train, docs_dev, y_dev, docs_test, y_test


def nltk_tokenizer_flat(x_raw, language):
    tokens_list = []
    for i in tqdm(range(0, len(x_raw))):
        tokens = nltk.word_tokenize(x_raw[i])
        tokens_doc = [word.lower() for word in tokens if all_punct(word) and
                      word not in stopwords.words(language) and num_there(word)]

        tokens_list.append(tokens_doc)

    return tokens_list


def load_data(data_path):
    x_train = []
    y_train = []
    x_dev = []
    y_dev = []
    x_test = []
    y_test = []

    train_name = data_path + "/train.txt"
    dev_name = data_path + "/dev.txt"
    test_name = data_path + "/test.txt"

    with open(train_name, 'r') as infile:
        reader = csv.reader(infile, delimiter=SEPARATOR, quotechar=None)
        for text, label in reader:
            x_train.append(text)
            y_train.append(label)

    with open(dev_name, 'r') as infile:
        reader = csv.reader(infile, delimiter=SEPARATOR, quotechar=None)
        for text, label in reader:
            x_dev.append(text)
            y_dev.append(label)

    with open(test_name, 'r') as infile:
        reader = csv.reader(infile, delimiter=SEPARATOR, quotechar=None)
        for text, label in reader:
            x_test.append(text)
            y_test.append(label)

    n_classes = len(set(list(y_train)))
    print("n_classes=", n_classes)

    return x_train, y_train, x_dev, y_dev, x_test, y_test


def read_cldc_docs_old(path, train_langs, test_langs):
    if "," in train_langs:
        train_lang = train_langs.split(",")
    else:
        train_lang = [train_langs]
    if "," in test_langs:
        test_lang = test_langs.split(",")
    else:
        test_lang = [test_langs]

    docs_train = {}
    label_train = {}
    docs_dev = {}
    label_dev = {}
    docs_test = {}
    label_test = {}
    for lang in list(set(train_lang+test_lang)):
        data_path = path + lang
        print("1. Loading data ....")
        x_train, y_train, x_dev, y_dev, x_test, y_test = load_data(data_path)

        print("2. Data Preprocessing ...")
        x_train_pro_p = data_path + "/x_train_pro.p"
        x_dev_pro_p = data_path + "/x_dev_pro.p"
        x_test_pro_p = data_path + "/x_test_pro.p"

        if not os.path.isfile(x_train_pro_p) or not os.path.isfile(x_dev_pro_p) or not os.path.isfile(x_test_pro_p):
            x_train_pro = nltk_tokenizer_flat(x_train)
            x_dev_pro = nltk_tokenizer_flat(x_dev)
            x_test_pro = nltk_tokenizer_flat(x_test)

            print("Saving to Pickle files ...")
            with open(x_train_pro_p, "wb") as file:
                pkl.dump(x_train_pro, file)

            with open(x_dev_pro_p, "wb") as file:
                pkl.dump(x_dev_pro, file)

            with open(x_test_pro_p, "wb") as file:
                pkl.dump(x_test_pro, file)
        else:
            print("Loading from Pickle files ...")
            with open(x_train_pro_p, "rb") as file:
                x_train_pro = pkl.load(file)

            with open(x_dev_pro_p, "rb") as file:
                x_dev_pro = pkl.load(file)

            with open(x_test_pro_p, "rb") as file:
                x_test_pro = pkl.load(file)

            print("Adding language prefix to the word")
            x_train_new = []
            for x_train in x_train_pro:
                x_train_new_sub = []
                for word in x_train:
                    x_train_new_sub.append(lang_dict[lang]+"_"+word)
                x_train_new.append(x_train_new_sub)

            x_dev_new = []
            for x_dev in x_dev_pro:
                x_dev_new_sub = []
                for word in x_dev:
                    x_dev_new_sub.append(lang_dict[lang]+"_"+word)
                x_dev_new.append(x_dev_new_sub)

            x_test_new = []
            for x_test in x_test_pro:
                x_test_new_sub = []
                for word in x_test:
                    x_test_new_sub.append(lang_dict[lang]+"_"+word)
                x_test_new.append(x_test_new_sub)

        if lang in train_lang:
            docs_train.update({lang: x_train_new})
            label_train.update({lang: y_train})
            docs_dev.update({lang: x_dev_new})
            label_dev.update({lang: y_dev})

        if lang in test_lang:
            docs_test.update({lang: x_test_new})
            label_test.update({lang: y_test})

    return docs_train, label_train, docs_dev, label_dev, docs_test, label_test


def split_docs(inputs_docs, vocab, lang):
    inputs_doc_split = []
    for doc in tqdm(inputs_docs):
        #max_sent = 0
        inputs_doc_split_sub = []
        for sent in doc:
            #max_words = 0
            sent_sub = []
            exclude = set(string.punctuation)
            new_sent = ''.join(ch for ch in sent if ch not in exclude)
            new_sent = ''.join([i for i in new_sent if not i.isdigit()]).lower()
            stop = set(stopwords.words(lang))
            for word in new_sent.split(" "):
                if word not in stop and len(word) > 2:
                    if lang_dict[lang] + "_" + word not in vocab:
                        vocab.update({lang_dict[lang] + "_" + word: len(vocab)})
                    if len(sent_sub) < 10:
                        sent_sub.append(vocab[lang_dict[lang] + "_" + word])
            """
            if len(sent_sub) > max_words:
                max_words = len(sent_sub)
            """
            if len(inputs_doc_split_sub) < 10 :
                inputs_doc_split_sub.append(sent_sub)
        """
        if len(inputs_doc_split_sub) > max_sent:
            max_sent = len(inputs_doc_split_sub)
        """

        inputs_doc_split.append(inputs_doc_split_sub)
    return inputs_doc_split, vocab


def read_par_sents(path, train_langs, test_langs):
    sent_src_train_dict = {}
    sent_trg_train_dict = {}
    sent_src_dev_dict = {}
    sent_trg_dev_dict = {}
    sent_src_test_dict = {}
    sent_trg_test_dict = {}
    langs = list(set(train_langs.split(",") + test_langs.split(",")))
    if "english" in langs:
        langs.remove("english")
    for lang in langs:
        print("Loading parallel sentences for english and " + lang)
        sent_src = []
        sent_trg = []
        with open(path + "20151028." + lang_dict[lang] + "__20151028.en") as file:
            lines = file.readlines()
            for i in tqdm(range(len(lines))):
                parts = lines[i].strip("").split(" ||| ")
                sent_src.append(parts[0])
                sent_trg.append(parts[1])

        random.shuffle(sent_src)
        random.shuffle(sent_trg)
        train_n = int(len(sent_src) * TRAIN_PERC)
        dev_n = int(len(sent_src) * DEV_PERC)
        train_src, train_trg, test_src, test_trg = sent_src[:train_n], sent_trg[:train_n], sent_src[train_n:], sent_trg[
                                                                                                               train_n:]
        dev_src, dev_trg, test_src, test_trg = test_src[:dev_n], test_trg[:dev_n], test_src[dev_n:], test_trg[dev_n:]

        sent_src_train_dict.update({lang: train_src})
        sent_trg_train_dict.update({lang: train_trg})
        sent_src_dev_dict.update({lang: dev_src})
        sent_trg_dev_dict.update({lang: dev_trg})
        sent_src_test_dict.update({lang: test_src})
        sent_trg_test_dict.update({lang: test_trg})

    return sent_src_train_dict, sent_trg_train_dict, sent_src_dev_dict, sent_trg_dev_dict, \
           sent_src_test_dict, sent_trg_test_dict, langs


def split_src_trg(inputs_sent, vocab, lang):
    inputs_split = []
    for sent in inputs_sent:
        sent_sub = []
        exclude = set(string.punctuation)
        new_sent = ''.join(ch for ch in sent if ch not in exclude)
        new_sent = ''.join([i for i in new_sent if not i.isdigit()]).lower()
        stop = set(stopwords.words(lang))
        for word in new_sent.split(" "):
            if word not in stop and len(word) > 2:
                if lang_dict[lang] + "_" + word not in vocab:
                    vocab.update({lang_dict[lang] + "_" + word: len(vocab)})
                sent_sub.append(vocab[lang_dict[lang] + "_" + word])
        inputs_split.append(sent_sub)

    return inputs_split, vocab


def load_fast_text(model_dir, language, vocab_list):
    with open(model_dir + "wiki." + lang_dict[language] + ".vec") as vector_file:
        word_vecs = vector_file.readlines()[1:]

    model = {}
    for word in tqdm(word_vecs):
        parts = word.split(" ")
        if lang_dict[language]+"_"+parts[0] in vocab_list:
            model.update({lang_dict[language]+"_"+parts[0]: map(float, parts[1:-1])})

    embed_dim = len(model[list(model.keys())[0]])

    return model, embed_dim


def load_embeddings(model_dir, model_file):
    model = {}
    with open(model_dir + model_file) as file_model:
        data = file_model.readlines()

    print("Loading list of words and their vectors in all languages ....")
    if model_file == "joint_emb_ferreira_2016_reg-l1_mu-1e-9_epochs-50" \
            or model_file == "multi_embed_linear_projection":
        for i in tqdm(range(0, len(data))):
            lang = data[i].split(" ")[0].split("_")[1]
            if lang in ["en", "fr", "de", "it"]:
                word = lang + "_" + data[i].split(" ")[0].split("_")[0]
                vectors = [float(vector) for vector in data[i].split(" ")[1:]]
                model.update({word: vectors})
    elif model_file == "semantic_spec_mrksic_2017-en_de_it_ru-ende-lang-joint-1e-09" \
            or model_file == "fasttext_en_de_fr_it.vec" or model_file == "unsupervised_fastext.txt" \
            or model_file == "supervised_fastext.txt" or model_file == "expert_dict_dim_red_en_de_fr_it.txt":
        print("Reading embeddings file:")
        for i in tqdm(range(0, len(data))):
            lang = data[i].split(" ")[0].split("_")[0]
            if lang in ["en", "fr"]:
                word = lang + "_" + data[i].split(" ")[0].split("_")[1]
                vectors = [float(vector) for vector in data[i].split(" ")[1:]]
                model.update({word: vectors})
    else:
        for i in tqdm(range(0, len(data))):
            lang = data[i].split(" ")[0].split(":")[0]
            if lang in ["en", "fr", "de", "it"]:
                word = lang + "_" + data[i].split(" ")[0].split(":")[1]
                vectors = [float(vector) for vector in data[i].split(" ")[1:]]
                model.update({word: vectors})

    embed_dim = len(model[list(model.keys())[0]])

    print("embed_dim=", embed_dim)
    return model, embed_dim


def build_embedding_matrix(vocab, model, embed_dim, vocab_dict):
    not_covered_words = []
    covered_words = []

    covered_count = 0
    not_covered_count = 0

    embedding_matrix = np.zeros((len(vocab), embed_dim))
    i = 0
    for word in vocab.keys():
        if word in model:
            embedding_vector = model[word]
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
            covered_words.append(word)
            covered_count += vocab_dict[word]
        else:
            not_covered_words.append(word)
            not_covered_count += vocab_dict[word]
        i += 1
    print("Number of Covered words ==>", covered_count)

    """
    print("Saving Covered Words ...")
    with open(save_path+"_covered_words.txt", "w") as file:
        for word in covered_words:
            file.write(word+"\n")

    """
    print("Number of Non-covered words ==>", not_covered_count)

    """
    print("Saving not-Covered Words ...")
    with open(save_path+"_not_covered_words.txt", "w") as file:
        for word in not_covered_words:
            file.write(word.encode("utf-8")+"\n")

    """
    return embedding_matrix


def doc_batch(inputs, batch_size, labels):
    #
    #  Shuffle the two lists: inputs, labels
    c = list(zip(inputs, labels))
    random.shuffle(c)
    inputs, labels = zip(*c)
    counts = collections.Counter(labels)
    weightsArray = []
    for i in range(len(list(set(labels)))):
        weightsArray.append(math.log(len(labels)/max(counts[i],1))+1)

    num_batches = int(len(inputs) / batch_size)
    batches = []
    for i in range(num_batches):
        inputs_batch = inputs[i * batch_size:(i + 1) * batch_size]
        labels_batch = labels[i * batch_size:(i + 1) * batch_size]
        document_sizes = np.array([len(doc) for doc in inputs_batch], dtype=np.int32)
        document_size = document_sizes.max()

        sentence_sizes_ = [[len(sent) for sent in doc] for doc in inputs_batch]
        sentence_size = max(map(max, sentence_sizes_))

        doc = np.zeros(shape=[batch_size, document_size, sentence_size], dtype=np.int32)  # == PAD
        l = np.zeros(shape=[batch_size, ], dtype=np.int32)
        w = np.zeros(shape=[batch_size, ], dtype=np.int32)

        sentence_sizes = np.zeros(shape=[batch_size, document_size], dtype=np.int32)
        for i, document in enumerate(inputs_batch):
            l[i] = labels_batch[i]
            w[i] = weightsArray[labels_batch[i]]
            for j, sentence in enumerate(document):
                sentence_sizes[i, j] = sentence_sizes_[i][j]
                for k, word in enumerate(sentence):
                    doc[i, j, k] = word

        batches.append({"docs": doc, "document_sizes": document_sizes, "sentence_sizes": sentence_sizes,
                        "labels": l, "weights": w})

    return batches


def doc_test_batch(inputs, labels):
    #
    #  Shuffle the two lists: inputs, labels
    c = list(zip(inputs, labels))
    random.shuffle(c)
    inputs, labels = zip(*c)

    document_sizes = np.array([len(doc) for doc in inputs], dtype=np.int32)
    document_size = document_sizes.max()

    sentence_sizes_ = [[len(sent) for sent in doc] for doc in inputs]
    sentence_size = max(map(max, sentence_sizes_))

    doc = np.zeros(shape=[len(inputs), document_size, sentence_size], dtype=np.int32)  # == PAD
    l = np.zeros(shape=[len(inputs), ], dtype=np.int32)
    w = np.zeros(shape=[len(inputs), ], dtype=np.int32)

    sentence_sizes = np.zeros(shape=[len(inputs), document_size], dtype=np.int32)
    for i, document in enumerate(inputs):
        l[i] = labels[i]
        w[i] = 1
        for j, sentence in enumerate(document):
            sentence_sizes[i, j] = sentence_sizes_[i][j]
            for k, word in enumerate(sentence):
                doc[i, j, k] = word

    doc_label = {"docs": doc, "document_sizes": document_sizes, "sentence_sizes": sentence_sizes,
                 "labels": l, "weights": w}

    return doc_label


def src_trg_batch(src_inputs, trg_inputs, batch_size):
    labels = [1] * len(src_inputs)
    #  Shuffle the two lists: src_inputs, trg_inputs
    c = list(zip(src_inputs, trg_inputs, labels))
    random.shuffle(c)
    src_inputs, trg_inputs, labels = zip(*c)

    num_batches = int(len(src_inputs) / batch_size)
    batches = []

    for i in range(num_batches):
        src_inputs_batch = src_inputs[i * batch_size:(i + 1) * batch_size]
        trg_inputs_batch = trg_inputs[i * batch_size:(i + 1) * batch_size]

        sent_src_sizes_ = np.array([len(sent) for sent in src_inputs], dtype=np.int32)
        sent_src_size = sent_src_sizes_.max()

        sent_trg_sizes_ = np.array([len(sent) for sent in trg_inputs], dtype=np.int32)
        sent_trg_size = sent_trg_sizes_.max()

        src = np.zeros(shape=[batch_size, sent_src_size], dtype=np.int32)  # == PAD
        trg = np.zeros(shape=[batch_size, sent_trg_size], dtype=np.int32)  # == PAD
        label = np.zeros(shape=[batch_size, 1], dtype=np.int32)

        sent_src_sizes = np.zeros(shape=[batch_size, 1], dtype=np.int32)
        sent_trg_sizes = np.zeros(shape=[batch_size, 1], dtype=np.int32)

        for i, src_trg in enumerate(zip(src_inputs_batch, trg_inputs_batch)):
            sent_src_sizes[i] = sent_src_sizes_[i]
            sent_trg_sizes[i] = sent_trg_sizes_[i]
            for j, word in enumerate(src_trg[0]):
                src[i, j] = word
            for j, word in enumerate(src_trg[1]):
                trg[i, j] = word
            label[i] = 1

        batches.append({"srcs": src, "sent_src_sizes": sent_src_sizes, "trgs": trg, "sent_trg_sizes": sent_trg_sizes,
                        "src_trg_labels": label})

    return batches


def src_trg_test_batch(src_inputs, trg_inputs):
    labels = [1] * len(src_inputs)
    #  Shuffle the two lists: src_inputs, trg_inputs, labels
    c = list(zip(src_inputs, trg_inputs, labels))
    random.shuffle(c)
    src_inputs, trg_inputs, labels = zip(*c)

    sent_src_sizes_ = np.array([len(sent) for sent in src_inputs], dtype=np.int32)
    sent_src_size = sent_src_sizes_.max()

    sent_trg_sizes_ = np.array([len(sent) for sent in trg_inputs], dtype=np.int32)
    sent_trg_size = sent_trg_sizes_.max()

    src = np.zeros(shape=[len(src_inputs), sent_src_size], dtype=np.int32)  # == PAD
    trg = np.zeros(shape=[len(src_inputs), sent_trg_size], dtype=np.int32)  # == PAD
    label = np.zeros(shape=[len(src_inputs), 1], dtype=np.int32)

    sent_src_sizes = np.zeros(shape=[len(src_inputs), 1], dtype=np.int32)
    sent_trg_sizes = np.zeros(shape=[len(src_inputs), 1], dtype=np.int32)

    for i, src_trg in enumerate(zip(src_inputs, trg_inputs, labels)):
        sent_src_sizes[i] = sent_src_sizes_[i]
        sent_trg_sizes[i] = sent_trg_sizes_[i]
        for j, word in enumerate(src_trg[0]):
            src[i, j] = word
        for j, word in enumerate(src_trg[1]):
            trg[i, j] = word
        label[i] = 1

    src_trg = {"srcs": src, "sent_src_sizes": sent_src_sizes, "trgs": trg, "sent_trg_sizes": sent_trg_sizes,
               "src_trg_labels": label}

    return src_trg
