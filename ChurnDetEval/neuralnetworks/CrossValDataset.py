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
import cPickle as pkl

def num_there(s):
    return not any(i.isdigit() for i in s)

def all_punct(s):
    return not all(i in punctuation for i in s)


class Object(object):
    pass


class CrossValDataset(object):
    def __init__(self, data_paths, train_mode, language, word_embeddings_path=None,
                 vocab_size=None, max_seq_length=None):
        self.data_paths = data_paths
        self.word_embeddings_path = word_embeddings_path
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.labels_to_ids = {}
        self.train_mode = train_mode
        self.language = language

        # STEP 1 => Load data from the file path given by the "dataset" argument
        self.x, self.y = self.__load_data(data_paths)

    def __load_data(self, data_paths, separator="\t"):
        # Initialization of lists containing the text and the labels
        x_dict = {}
        y_dict = {}

        for lang in data_paths:
            print("lang=", lang)
            x = []
            y = []
            with open(data_paths[lang], 'r') as infile:
                reader = csv.reader(infile, delimiter=separator, quotechar=None)
                print(reader)
                for text, label in reader:
                    x.append(text)
                    y.append(label)
            x_dict.update({lang: x})
            y_dict.update({lang: y})

        return x_dict, y_dict

    def build_embedding_matrix(self, vocab):
        print("Saving Churn Vocabulary")
        with open("/aimlx/Results/ChurnDet/vocab_churn.p", "wb") as file:
            pkl.dump(vocab, file)

        model = {}
        if "mono" in self.train_mode:
            """
            if self.language in ["en", "de"]:
                model_gensim = KeyedVectors.load_word2vec_format(self.word_embeddings_path, binary=True)
                for word in model_gensim.wv.vocab:
                    model.update({self.language+"_"+word: model_gensim[word]})
            else:
            """
            with open(self.word_embeddings_path) as vector_file:
                word_vecs = vector_file.readlines()[1:] #[1:]

            model = {}
            for word in tqdm(word_vecs):
                parts = word.strip(" \n").split(" ")
                vectors = [float(vector) for vector in parts[1:]]
                model.update({self.language+"_"+parts[0]: vectors})
        else:
            with open(self.word_embeddings_path) as model_file:
                data = model_file.readlines()

            print("Loading list of words and their vectors in all languages ....")

            #if "semantic" in self.word_embeddings_path or "fasttext" in self.word_embeddings_path:
            for i in tqdm(range(0, len(data))):
                lang = data[i].split(" ")[0].split("_")[0]
                word = data[i].split(" ")[0].split("_")[1]
                if lang in ["en", "de"]:
                    if False:#if lang == "de":
                        word = lang + "_" + word.decode("utf-8")
                    else:
                        word = lang + "_" + word
                    if word in vocab:
                        vectors = [float(vector) for vector in data[i].split(" ")[1:]]
                        model.update({word: vectors})# model.update({lang+"_"+word: vectors})
            """
            else:
                for i in tqdm(range(0, len(data))):
                    lang = data[i].split(" ")[0].split(":")[0]
                    if lang in ["en", "fr", "de", "it"]:
                        word = lang + "_" + data[i].split(" ")[0].split(":")[1]
                        vectors = [float(vector) for vector in data[i].split(" ")[1:]]
                        model.update({word: vectors})# model.update({lang+"_"+word: vectors})
                        
            """
        print("len(model)=", len(model))
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
        :param lang:
        :param x_raw:
        :return: tokens_list: the list of tokens (no lemmatization) per each sentence
        """

        tokens_list = []  # List of list of tokens
        for i in tqdm(range(0, len(x_raw))):
            #tokens = x_raw[i].split(" ")
            tokens = nltk.word_tokenize(x_raw[i].decode('utf-8'))
            tokens_doc = [lang + "_" + word.lower() for word in tokens if all_punct(word) and num_there(word)]
            #tokens_doc = [lang + "_" + word.lower() for word in tokens]
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
    def convert_ids_pad(self, x_pro, vocab, add_pad):
        sequences_dict = {}
        for lang in x_pro:
            sequences = []
            for doc in x_pro[lang]:
                list_ids_sub = []
                for token in doc:
                    list_ids_sub.append(vocab[token])
                sequences.append(list_ids_sub)
            x_padded = pad_sequences(sequences, padding='post', maxlen=self.max_seq_length, value=add_pad)
            sequences_dict.update({lang: x_padded})

        return sequences_dict

    def prepare_data(self):
        x = self.x
        y = self.y

        # Convert labels to integer
        labels_dict = {}
        for lang in y:
            labels = []
            for label in y[lang]:
                if label not in self.labels_to_ids.keys():
                    self.labels_to_ids[label] = len(self.labels_to_ids)
                label = self.labels_to_ids[label]
                labels.append(label)
            labels_dict.update({lang: labels})

        for lang in x:
            print('Found %s datapoints for training data for {} is {}.'. format(lang, len(x[lang])))

        print("Tokenizing and indexing data...")

        x_all = []
        texts_tok = {}
        for lang in x:
            tokenized = self.nltk_tokenizer(lang, x[lang])
            texts_tok.update({lang: tokenized})
            x_all += tokenized

        print(x_all[0])

        print("Build the vocabulary")
        vocab = self.create_vocabulary(x_all)
        print('Found %s unique tokens.' % len(vocab))

        print("Converting to ids")
        x_dict = self.convert_ids_pad(texts_tok, vocab, len(vocab)+1)

        y_dict = {}
        for lang in labels_dict:
            y_dict.update({lang: to_categorical(np.asarray(labels_dict[lang]))})

        embedding_matrix = self.build_embedding_matrix(vocab)

        return x_dict, labels_dict, y_dict, embedding_matrix, len(embedding_matrix), self.labels_to_ids, vocab

    def prepare_data_bot(self, vocab, labels_to_ids):
        x = self.x
        y = self.y

        # Convert labels to integer
        labels_dict = {}
        for lang in y:
            labels = []
            for label in y[lang]:
                label = labels_to_ids[label]
                labels.append(label)
            labels_dict.update({lang: labels})

        for lang in x:
            print('Found %s datapoints for training data for {} is {}.'. format(lang, len(x[lang])))

        print("Tokenizing data...")

        x_all = []
        texts_tok = {}
        for lang in x:
            tokenized = self.nltk_tokenizer(lang, x[lang])
            texts_tok.update({lang: tokenized})
            x_all += tokenized

        words_set = list(set([item for sublist in x_all for item in sublist]))

        print("Converting to ids")
        pad_add = len(vocab)+1
        for word in words_set:
            if word not in vocab:
                vocab.update({word: pad_add})
                #pad_add = pad_add + 1

        x_dict = self.convert_ids_pad(texts_tok, vocab, pad_add)

        y_dict = {}
        for lang in labels_dict:
            y_dict.update({lang: to_categorical(np.asarray(labels_dict[lang]))})

        #embedding_matrix = self.build_embedding_matrix(vocab)

        return x_dict, y_dict #, embedding_matrix, vocab, len(vocab) +1
