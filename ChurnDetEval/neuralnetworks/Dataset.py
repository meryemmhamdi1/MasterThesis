# -*- coding: utf-8 -*-

# This script convert the data to the right format for the neuralnetworks

import csv
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from gensim.models import KeyedVectors
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from string import punctuation

def num_there(s):
    return not any(i.isdigit() for i in s)

def all_punct(s):
    return not all(i in punctuation for i in s)


class Object(object):
    pass


class Dataset(object):
    def __init__(self, train_set, dev_set, test_set, train_mode, language, word_embeddings_path=None,
                 vocab_size=None, max_seq_length=None):
        self.train_set = train_set
        self.dev_set = dev_set
        self.test_set = test_set
        self.word_embeddings_path = word_embeddings_path
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.labels_to_ids = {}
        self.train_mode = train_mode
        self.language = language

        # STEP 1 => Load data from the file path given by the "dataset" argument
        self.x_train, self.y_train, self.x_dev, self.y_dev, self.x_test, self.y_test = \
            self.__load_data(train_set, dev_set, test_set)

    def __load_data(self, train_name, dev_name, test_name, separator="\t"):
        # Initialization of lists containing the text and the labels
        x_train_dict = {}
        y_train_dict = {}
        x_dev_dict = {}
        y_dev_dict = {}
        x_test_dict = {}
        y_test_dict = {}

        for train_lang in train_name:
            x_train = []
            y_train = []
            with open(train_name[train_lang], 'r') as infile:
                reader = csv.reader(infile, delimiter=separator, quotechar=None)
                for text, label in reader:
                    x_train.append(text)
                    y_train.append(label)
            x_train_dict.update({train_lang: x_train})
            y_train_dict.update({train_lang: y_train})

        for dev_lang in dev_name:
            x_dev = []
            y_dev = []
            with open(dev_name[dev_lang], 'r') as infile:
                reader = csv.reader(infile, delimiter=separator, quotechar=None)
                for text, label in reader:
                    x_dev.append(text)
                    y_dev.append(label)
            x_dev_dict.update({dev_lang: x_dev})
            y_dev_dict.update({dev_lang: y_dev})

        for test_lang in test_name:
            x_test = []
            y_test = []
            with open(test_name[test_lang], 'r') as infile:
                reader = csv.reader(infile, delimiter=separator, quotechar=None)
                for text, label in reader:
                    x_test.append(text)
                    y_test.append(label)
            x_test_dict.update({test_lang: x_test})
            y_test_dict.update({test_lang: y_test})

        return x_train_dict, y_train_dict, x_dev_dict, y_dev_dict, x_test_dict, y_test_dict

    def build_embedding_matrix(self, vocab):
        model = {}
        if "mono" in self.train_mode:
            if self.language in ["en", "de"]:
                model_gensim = KeyedVectors.load_word2vec_format(self.word_embeddings_path, binary=True)
                for word in model_gensim.wv.vocab:
                    model.update({self.language+"_"+word: model_gensim[word]})
            else:
                with open(self.word_embeddings_path) as vector_file:
                    word_vecs = vector_file.readlines()[1:]

                model = {}
                for word in word_vecs:
                    parts = word.split(" ")
                    model.update({self.language+"_"+parts[0]: map(float, parts[1:])})
        else:
            with open(self.word_embeddings_path) as model_file:
                data = model_file.readlines()

            print("Loading list of words and their vectors in all languages ....")

            if "final" in self.word_embeddings_path or "semantic" in self.word_embeddings_path
                or "joint" in self.word_embeddings_path:
                for i in tqdm(range(0, len(data))):
                    lang = data[i].split(" ")[0].split("_")[0]
                    if lang in ["en", "fr", "de", "it"]:
                        word = data[i].split(" ")[0]
                        vectors = [float(vector) for vector in data[i].split(" ")[1:]]
                        model.update({word: vectors})# model.update({lang+"_"+word: vectors})
            else:
                for i in tqdm(range(0, len(data))):
                    lang = data[i].split(" ")[0].split(":")[0]
                    if lang in ["en", "fr", "de", "it"]:
                        word = lang + "_" + data[i].split(" ")[0].split(":")[1]
                        vectors = [float(vector) for vector in data[i].split(" ")[1:]]
                        model.update({word: vectors})# model.update({lang+"_"+word: vectors})

        embed_dim = len(model[model.keys()[0]])
        not_covered_words = []
        covered_words = []
        embedding_matrix = np.zeros((len(vocab) + 1, embed_dim))
        i = 0
        for word in vocab.keys():
            if word in model:
                embedding_vector = model[word]
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
                covered_words.append(word)
            else:
                not_covered_words.append(word)
            i += 1
        print("Covered words ==>", len(covered_words))
        print("Non-covered words ==>", len(not_covered_words))
        return embedding_matrix

    def nltk_tokenizer(self, lang, x_raw):
        """
        :param mode: specifies whether it is train, validation or test part of the data to be tokenized
        :return: tokens_list: the list of tokens (no lemmatization) per each sentence
        """
        tokens_list = []  # List of list of tokens
        for i in tqdm(range(0, len(x_raw))):
            tokens = nltk.word_tokenize(x_raw[i].decode('utf-8'))
            tokens_doc = [lang + "_" + word.lower() for word in tokens if all_punct(word) and
                          word not in stopwords.words("french") and num_there(word)]

            tokens_list.append(tokens_doc)

        return tokens_list

    def create_vocabulary(self, x_all):
        vocab_dict = {}
        for doc in x_all:
            for token in doc:
                if token in vocab_dict:
                    vocab_dict[token] += 1
                else:
                    vocab_dict[token] = 1
        vocab_list = sorted(vocab_dict)
        vocab = dict([x,y] for (y, x) in enumerate(vocab_list))

        return vocab

    """ Converting to ids and pad to fixed length """
    def convert_ids_pad(self, x_train_pro, x_dev_pro, x_test_pro, vocab):
        sequences_train_dict = {}
        for lang in x_train_pro:
            sequences_train = []
            for doc in x_train_pro[lang]:
                list_ids_sub = []
                for token in doc:
                    list_ids_sub.append(vocab[token])
                sequences_train.append(list_ids_sub)
            train_padded = pad_sequences(sequences_train, padding='post', maxlen=self.max_seq_length, value=len(vocab)+1)
            sequences_train_dict.update({lang: train_padded})

        sequences_dev_dict = {}
        for lang in x_dev_pro:
            sequences_dev = []
            for doc in x_dev_pro[lang]:
                list_ids_sub = []
                for token in doc:
                    list_ids_sub.append(vocab[token])
                sequences_dev.append(list_ids_sub)
            dev_padded = pad_sequences(sequences_dev, padding='post', maxlen=self.max_seq_length, value=len(vocab)+1)
            sequences_dev_dict.update({lang: dev_padded})

        sequences_test_dict = {}
        for lang in x_test_pro:
            sequences_test = []
            for doc in x_test_pro[lang]:
                list_ids_sub = []
                for token in doc:
                    list_ids_sub.append(vocab[token])
                sequences_test.append(list_ids_sub)
            test_padded = pad_sequences(sequences_test, padding='post', maxlen=self.max_seq_length, value=len(vocab)+1)
            sequences_test_dict.update({lang: test_padded})

        return sequences_train_dict, sequences_dev_dict, sequences_test_dict

    def prepare_data(self):
        train_texts = self.x_train
        y_train = self.y_train
        dev_texts = self.x_dev
        y_dev = self.y_dev
        test_texts = self.x_test
        y_test = self.y_test

        # Convert labels to integer
        train_labels_dict = {}
        for train_lang in y_train:
            train_labels = []
            for label in y_train[train_lang]:
                if label not in self.labels_to_ids.keys():
                    self.labels_to_ids[label] = len(self.labels_to_ids)
                label = self.labels_to_ids[label]
                train_labels.append(label)
            train_labels_dict.update({train_lang: train_labels})

        dev_labels_dict = {}
        for dev_lang in y_dev:
            dev_labels = []
            for label in y_dev[dev_lang]:
                label = self.labels_to_ids[label]
                dev_labels.append(label)
            dev_labels_dict.update({dev_lang: dev_labels})

        test_labels_dict = {}
        for test_lang in y_test:
            test_labels = []
            for label in y_test[test_lang]:
                label = self.labels_to_ids[label]
                test_labels.append(label)
            test_labels_dict.update({test_lang: test_labels})

        for lang in train_texts:
            print('Found %s datapoints for training data for {} is {}.'. format(lang, len(train_texts[lang])))

        for lang in dev_texts:
            print('Found %s datapoints for dev data for {} is {}.'.format(lang, len(dev_texts[lang])))

        for lang in test_texts:
            print('Found %s datapoints for testing data for {} is {}.'.format(lang,len(test_texts[lang])))

        print("Tokenizing and indexing data...")

        x_all = []
        train_texts_tok = {}
        for lang in train_texts:
            tokenized = self.nltk_tokenizer(lang, train_texts[lang])
            train_texts_tok.update({lang: tokenized})
            x_all += tokenized

        dev_texts_tok = {}
        for lang in dev_texts:
            tokenized = self.nltk_tokenizer(lang, dev_texts[lang])
            dev_texts_tok.update({lang: tokenized})
            x_all += tokenized

        test_texts_tok = {}
        for lang in test_texts:
            tokenized = self.nltk_tokenizer(lang, test_texts[lang])
            test_texts_tok.update({lang: tokenized})
            x_all += tokenized

        print(x_all[0])

        print ("Build the vocabulary")
        vocab = self.create_vocabulary(x_all)
        print('Found %s unique tokens.' % len(vocab))

        print ("Converting to ids")
        x_train_dict, x_dev_dict, x_test_dict = \
            self.convert_ids_pad(train_texts_tok, dev_texts_tok, test_texts_tok, vocab)

        y_train_dict = {}
        for lang in x_train_dict:
            y_train_dict.update({lang: to_categorical(np.asarray(train_labels_dict[lang]))})

        y_dev_dict = {}
        for lang in x_dev_dict:
            y_dev_dict.update({lang: to_categorical(np.asarray(dev_labels_dict[lang]))})

        y_test_dict = {}
        for lang in x_test_dict:
            y_test_dict.update({lang: to_categorical(np.asarray(test_labels_dict[lang]))})

        embedding_matrix = self.build_embedding_matrix(vocab)

        return x_train_dict, y_train_dict, x_dev_dict, y_dev_dict, x_test_dict, y_test_dict, embedding_matrix, \
               len(embedding_matrix), self.labels_to_ids
