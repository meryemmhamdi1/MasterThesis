# coding: iso-8859-1
"""
This code defines different functions for NLP preprocessing including applying:
  * Tokenization
  * POS tagging for lemmatization, stopword removal and nava word extraction
  * Creation of Vocabulary and convertion from word to id
  * Word Vector representation using multlingual embeddings or any specified model
  * Creation and randomization of batches

Created on Tue Feb 27 2018

@author: Meryem M'hamdi (Meryem.Mhamdi@epfl.ch)
"""
from __future__ import absolute_import
try:
    import cPickle as pkl
except:
    import _pickle as pkl
import json
import os
import random as rnd
from string import punctuation

import nltk
import numpy as np
from nltk.corpus import stopwords
from tqdm import tqdm
from keras.preprocessing.text import Tokenizer
from nltk.tokenize import RegexpTokenizer

class MultiLangVector:

    def __init__(self, language, word, vector):
        self.language = language
        self.word = word
        self.vector = vector

    def toString(self):
        return "Language = "+self.language + " Word= "+self.word + "Vector= "+self.vector

    def getLanguage(self):
        return self.language

    def setLanguage(self, language):
        self.language = language

    def getWord(self):
        return self.word

    def setWord(self, word):
        self.word = word

    def getVector(self):
        return self.vector

    def setVector(self, vector):
        self.vector = vector

# Special vocabulary symbols
_PAD = "_PAD"
_UNK = "_UNK"
_BOS = "_BOS"
_EOS = "_EOS"
_START_VOCAB = [_PAD, _UNK]

def num_there(s):
    return not any(i.isdigit() for i in s)

def all_punct(s):
    return not all(i in punctuation for i in s)

def process_json(file_):
    # Clean json File by replacing problematic strings (if necessary)

    # Process the json data elements
    print("Processing json file ...")
    with open(file_) as data_file:
        data = json.load(data_file)
    # if len(data["X_ids"]) > 50000:
    #     max_ = 50000
    # else:
    #     max_ = len(data["X_ids"])
    # return data["X_ids"][0:max_], data["Y_ids"][0:max_]
    return data["X_ids"], data["Y_ids"]


class DataUtils(object):
    def __init__(self, data_root, pre_dir, stop_pos_path, lemma_use, stop_use, language, emb_model_path, emb_model_name,
                 embed_dim):
        self.data_root = data_root
        self.data_dir = data_root + language + "/"
        self.pre_dir = self.data_dir + pre_dir
        self.language = language
        self.emb_model_path = emb_model_path
        self.emb_model_name = emb_model_name
        self.stop_pos_path = stop_pos_path
        self.use_lemma = lemma_use
        self.use_stop = stop_use
        self.embed_dim = embed_dim

    def nltk_tokenizer_flat(self, x_raw):
        """
        :param mode: specifies whether it is train, validation or test part of the data to be tokenized
        :return: tokens_list: the list of tokens (no lemmatization) per each sentence
        """
        tokens_list = []  # List of list of tokens
        tokenizer = RegexpTokenizer("[\w']+")
        for i in tqdm(range(0, len(x_raw))):
            ## Tokenizing each sentence in doc
            #tokens = nltk.word_tokenize(x_raw[i].decode('utf-8'))
            tokens = tokenizer.tokenize(x_raw[i])#.encode("utf-8"))
            #tokens_doc = [word.lower() for word in tokens if all_punct(word) and word not in stopwords.words(self.language) and num_there(word)]

            tokens_list.append(tokens)

        return tokens_list

    def nltk_tokenizer(self, x_raw, mode):
        """

        :param mode: specifies whether it is train, validation or test part of the data to be tokenized
        :return: tokens_list: the list of tokens (no lemmatization) per each sentence
        """
        tokens_list = []  # List of list of tokens
        for i in tqdm(range(0, len(x_raw))):
            ## Tokenizing each sentence in doc
            tokens_list_sub = []
            for sent in x_raw[i]:
                tokens = nltk.word_tokenize(sent)
                tokens_sent = [word.lower() for word in tokens if word not in punctuation and
                               word not in stopwords.words(self.language)]
                if len(tokens_sent) > 0:
                    tokens_list_sub.append(tokens_sent)
            tokens_list.append(tokens_list_sub)

        return tokens_list

    ### Second Iteration
    def pos_tagging(self, tokenized_doc):
        """
        POS tagging of documents using universal tagset
        :param tokenized_doc:
        :return: tagged_doc
        """
        tagged_doc = []
        for i in tqdm(range(0, len(tokenized_doc))):
            tagged_doc.append(nltk.pos_tag(tokenized_doc[i]))
        return tagged_doc

    ### Second Iteration
    def normalize_pos_tags_words(self, tagged_doc):
        """
        Categorizing Penn Tags to Noun, Verb, Adjective, Adverb for easy extraction of NAVA words
        :param tagged_dialogues:
        :return: dialogues_nava
        """
        doc_nava = []
        doc_nava_sub = []
        for i in range(0, len(tagged_doc)):
            for (word, tag) in tagged_doc[i]:
                if tag == 'NN' or tag == 'NNP' or tag == 'NNPS' or tag == 'NNS':
                    doc_nava_sub.append((word, 'n'))
                elif tag == 'VB' or tag == 'VBD' or tag == 'VBG' or tag == 'VBN' or tag == 'VBP' or tag == 'VBZ':
                    doc_nava_sub.append((word, 'v'))
                elif tag == 'JJ' or tag == 'JJR' or tag == 'JJS':
                    doc_nava_sub.append((word, 'Adj'))
                elif tag == 'RB' or tag == 'RBR' or tag == 'RBS':
                    doc_nava_sub.append((word, 'Adv'))
                else:
                    doc_nava_sub.append((word, tag))
            doc_nava.append(list(doc_nava_sub))
            doc_nava_sub = []
        return doc_nava

    ### Second Iteration
    def keep_only_nava_words(self, tagged_doc):
        """
        :param tagged_dialogues:
        :return: dialogues_nava
        """
        doc_nava = []
        doc_nava_sub = []
        for i in range(0, len(tagged_doc)):
            for (word, tag) in tagged_doc[i]:
                if tag == "n" or tag == "v" or tag == "ADJ" or tag == "ADV":
                    doc_nava_sub.append(word)
            doc_nava.append(list(doc_nava_sub))
            doc_nava_sub = []
        return doc_nava

    ### Second Iteration
    def lemmatizer(self, tagged_doc):
        """
        Lemmatizing the document text using Word Net Lemmatizer using pos tags information
        :param documents: tagged text
        :return: lemmatized documents
        """
        doc_lemmatized = []
        lmtzr = WordNetLemmatizer()
        for i in range(0, len(tagged_doc)):
            doc_sub = []
            for (word, tag) in tagged_doc[i]:
                if tag == 'v' or tag == 'n':
                    doc_sub.append((lmtzr.lemmatize(word, tag), tag))
                else:
                    doc_sub.append((word, tag))
            doc_lemmatized.append(doc_sub)
        return doc_lemmatized

    ### Second Iteration
    def eliminate_stop_words_punct(self, tok_doc):
        """
        Elimination of Stop words using the language list
        Elimination of Punctuation
        :rtype: object
        :param tok_doc:
        :return: tok_doc_without
        """
        stop_words = stopwords.words(self.language)
        tok_doc_without = []
        kept_indices = []
        for i in range(0, len(tok_doc)):
            tok_doc_without_sub = []
            for word in tok_doc[i]:
                if word not in stop_words and word not in punctuation and len(word) >= 2:
                    tok_doc_without_sub.append((word.lower(), tag))
            if len(tok_doc_without_sub) > 0:
                tok_doc_without.append(tok_doc_without_sub)
                kept_indices.append(i)

        return tok_doc_without, kept_indices

    def load_vocab(self, vocab_path):
        rev_vocab = []
        with open(vocab_path, "r") as vocab_file:
            rev_vocab.extend(vocab_file.readlines())
        rev_vocab = [l.strip() for l in rev_vocab]
        vocab = dict([x, y] for (y, x) in enumerate(rev_vocab))
        return rev_vocab, vocab

    def create_vocab(self, vocab_path, list_):
        vocab_dict = {}
        for token in list_:
            if token in vocab_dict:
                vocab_dict[token] += 1
            else:
                vocab_dict[token] = 1

        vocab_list = sorted(vocab_dict)

        if not os.path.isfile(vocab_path + ".txt"):
            print("Creating vocabulary at %s from  " % vocab_path)
            with open(vocab_path, "w") as output_file:
                for token in vocab_list:
                    output_file.write(token + '\n')

        vocab = dict([x, y] for (y, x) in enumerate(vocab_list))

        with open(vocab_path + ".p", "wb") as file:
            pkl.dump(vocab, file)

        return vocab_list, vocab

    def create_vocab(self, text_lists):
        vocab_dict = {}
        for doc in text_lists:
            for token in doc:
                if token in vocab_dict:
                    vocab_dict[token] +=1
                else:
                    vocab_dict[token] = 1
        vocab_list = sorted(vocab_dict)
        vocab = dict([x, y] for (y, x) in enumerate(vocab_list))
        return vocab

    def label_to_ids_simple(self, y, label_vocab):
        label_ids = []
        for label in y:
            label_ids.append(label_vocab[label])

        return label_ids

    def label_to_ids(self, vocab_path, mode):
        label_ids_path = self.pre_dir + mode + "/" + self.language + "_label_ids.txt"
        label_input_path = self.pre_dir + mode + "/" + self.language + "_label.txt"
        if not os.path.isfile(label_ids_path):
            print ("Transforming to ids ")
            # Loading the vocabulary
            rev_vocab, vocab = self.load_vocab(vocab_path)
            # Reading tokens in the input path
            label_to_ids_list = []
            with open(label_input_path, 'r') as input_file:
                with open(label_ids_path, 'w') as tokens_file:
                    for i in range(0, len(input_file)):
                        token_id = vocab.get(input_file[i], 0)
                        label_to_ids_list.append(token_id)
                        tokens_file.write(token_id + "\n")

        return label_to_ids_list

    def load_w2v_model_map_ids(self, mode):
        pre_path = self.pre_dir + mode + "/"
        if not os.path.isfile(pre_path + self.language + "_x_words.p") or not os.path.isfile(
                                pre_path + self.language + "_y_labels.p"):
            # if not os.path.isfile(pre_path+self.language+"_x_words.p") or  not os.path.isfile(pre_path+self.language+"_y_labels.p"):
            with open(self.emb_model_path + self.language + ".pkl") as w2v_pkl_file:
                word_vectors = pkl.load(w2v_pkl_file)

            mapping_dict = {}
            for i in tqdm(range(0, len(word_vectors[0]))):
                mapping_dict.update({i: word_vectors[0][i]})

            data_file = self.data_dir + mode + "/" + self.language + ".json"

            x_raw, y_raw = process_json(data_file)

            print("Mapping x ids to words")
            x_conv = []
            for i in tqdm(range(0, len(x_raw))):  # for document
                x_conv_doc = []
                for sent in x_raw[i]:  # for sentence
                    x_conv_sent = []
                    for word in sent:
                        x_conv_sent.append(mapping_dict[word])
                    x_conv_doc.append(x_conv_sent)
                x_conv.append(x_conv_doc)

            if not os.path.isdir(pre_path):
                os.makedirs(pre_path)

            print("Saving x word and y words to pickle files")
            with open(pre_path + self.language + "_x_words.p", "wb") as x_words_pkl:
                pkl.dump(x_conv, x_words_pkl)

            with open(pre_path + self.language + "_y_labels.p", "wb") as y_words_pkl:
                pkl.dump(y_raw, y_words_pkl)
        else:
            with open(pre_path + self.language + "_x_words.p", "rb") as x_words_pkl:
                x_conv = pkl.load(x_words_pkl)

            with open(pre_path + self.language + "_y_labels.p", "rb") as y_words_pkl:
                y_raw = pkl.load(y_words_pkl)

        return x_conv, y_raw

    def load_multi_vectors(self):
        print("Executing load_multi_vectors function")
        with open(self.emb_model_path + self.emb_model_name) as model_file:
            data = model_file.readlines()

        lang_vectors_list = []
        language_set = set()

        print("Loading list of words and their vectors in all languages ....")
        for i in tqdm(range(0, len(data))):
            lang = data[i].split(" ")[0].split(":")[0]
            if lang in ["en", "fr", "de", "it"]:
                language_set.add(lang)
                word = data[i].split(" ")[0].split(":")[1]
                vectors = [float(vector) for vector in data[i].split(" ")[1:]]
                lang_vectors_list.append(MultiLangVector(lang, word, vectors))

        # Create word vectors for each language in a separate dictionary file:
        print("Creating word vectors for each language in a separate dictionary file ....")
        dictionaries_lang = {}
        for i in tqdm(range(0, len(lang_vectors_list))):
            lang = lang_vectors_list[i].getLanguage()
            word = lang_vectors_list[i].getWord()
            vector = lang_vectors_list[i].getVector()
            if lang in dictionaries_lang:
                # Update the dictionary by adding the new mapping between the word and the vector
                word_vectors_dict = dictionaries_lang.get(lang)
                word_vectors_dict.update({word: vector})
            else:
                word_vectors_dict = {word: vector}
                dictionaries_lang[lang] = word_vectors_dict

        # Read language ISO abbreviations
        abbr_dict = {}
        with open("iso_lang_abbr.txt") as iso_abbr_file:
            for line in iso_abbr_file:
                abbr_dict.update({line.split(':')[0]: line.split(':')[1][:-1]})

        pre_path = self.emb_model_path + self.emb_model_name + "_languages/"
        if not os.path.isdir(pre_path):
            os.makedirs(pre_path)
        # vectors for each language in a separate dictionary file:
        print("Save word vectors for each language in a separate dictionary file ....")
        for lang in dictionaries_lang:
            with open(pre_path + abbr_dict[lang] + "_vector_model.p", "wb") as vector_model_pkl:
                pkl.dump(dictionaries_lang.get(lang), vector_model_pkl)

    def apply_word2vec_gensim(self, tokens_list, model):
        vectors_list_doc = []
        for i in tqdm(range(0, len(tokens_list))):
            vectors_list_sent = []
            for sent in tokens_list[i]:
                vectors_sub_word = []
                if len(sent) > 0:
                    for token in sent:
                        # if token not in stopwords and token not in punct:
                        if token in model:
                            vectors_sub_word.append(model[token])
                        else:
                            zeros = list(np.zeros(self.embed_dim))
                            vectors_sub_word.append(zeros)
                    vectors_list_sent.append(np.mean(vectors_sub_word, 0))
            # if len(vectors_list_sent) > 1:
            #     vectors_list_doc.append(list(np.mean(vectors_list_sent, axis=0)))
            # else:
            #     vectors_list_doc.append([])
            vectors_list_doc.append(list(vectors_list_sent))
        return vectors_list_doc

    def apply_emb_model(self, tokens_list, word_vector_dict):

        # Iterate over the list of words in the token list
        print ("Generating the mappings from word to their multilingual vectors")
        vectors_list_doc = []
        for i in tqdm(range(0, len(tokens_list))):
            vectors_list_sent = []
            for sent in tokens_list[i]:
                vectors_sub_word = []
                if len(sent) > 0:
                    for token in sent:
                        if token in word_vector_dict:
                            vectors_sub_word.append(word_vector_dict[token])
                        else:
                            zeros = list(np.zeros(self.embed_dim))
                            vectors_sub_word.append(zeros)
                    vectors_list_sent.append(np.mean(vectors_sub_word, 0))
            # if len(vectors_list_sent) > 1:
            #     vectors_list_doc.append(list(np.mean(vectors_list_sent, axis=0)))
            # else:
            #     vectors_list_doc.append([])
            vectors_list_doc.append(list(vectors_list_sent))
        return vectors_list_doc

    def create_id_mapping(self, y, dict_map_path):
        print ("Creating id mappings")
        unique_list = list(set(tuple(i) for i in y))
        dict_map = {}
        for j in tqdm(range(0, len(y))):
            index_ = unique_list.index(tuple(y[j]))
            dict_map.update({tuple(y[j]): index_})

        ## Save the mapping somewhere
        with open(dict_map_path, "wb") as dict_pkl:
            pkl.dump(dict_map, dict_pkl)

        return dict_map

    def label_to_ids_1(self, y, dict_map):
        print("Converting a list of lists of integers to list of tuples, then creating a distinct vocabulary out "
              "of it to map them to list of ids")

        ## Upload the dictionary from pickle file
        y_ids = []
        for i in tqdm(range(0, len(y))):
            y_ids.append(dict_map[tuple(y[i])])

        # print(y_ids[0:10])
        return y_ids

    def sent_avg(self, x):
        new_docs = []
        for i in tqdm(range(0, len(x))):
            mean_np = np.mean(np.array(x[i]), axis=0)
            new_docs.append(list(mean_np))
        return np.array(new_docs)

    def pad_to_fixed_size(self, x, max_sentences):
        new_docs = []
        for i in tqdm(range(0, len(x))):
            if len(x[i]) < max_sentences:
                num_add = max_sentences - len(x[i])
                to_add = []
                for _ in range(0, num_add):
                    zeros_np = list(np.zeros(self.embed_dim))
                    to_add.append(zeros_np)
                new_docs.append(x[i] + to_add)
            else:
                new_docs.append(x[i][:max_sentences])
        return np.array(new_docs)

    def randomize_data(self, in_seq_train, out_seq_train, label_train, batch_size):
        new_size = int(len(in_seq_train) / batch_size) * batch_size
        indices = list(range(new_size))
        rnd.shuffle(indices)
        in_seq_train_new = []
        out_seq_train_new = []
        label_train_new = []
        for index in indices:
            in_seq_train_new.append(in_seq_train[index])
            out_seq_train_new.append(out_seq_train[index])
            label_train_new.append(label_train[index])

        return in_seq_train_new, out_seq_train_new, label_train_new

    def get_batch(self, in_seq_train, out_seq_train, label_train, batch_size, max_size, start):
        """

        :param in_seq_train: the list of input data after tokenization and tranformation from words to ids
        :param out_seq_train: the list of output slots ids
        :param label_train: the list of output intent labels
        :param batch_size:
        :param max_size:
        :return:
        """
        ## Choosing Random samples from the dataset that consists of source ids, slot target ids and intent ids

        tmp_encoder_inputs, tmp_intent_decoder_inputs, tmp_slot_decoder_inputs, \
        encoder_inputs, intent_decoder_inputs, slot_decoder_inputs = [], [], [], [], [], []

        # rnd_indices = []
        #
        # for _ in xrange(batch_size):
        #     rnd_indices.append(rnd.choice(range(len(in_seq_train))))

        for i in range(start, start + batch_size):
            tmp_encoder_input = in_seq_train[i]
            tmp_intent_decoder_input = label_train[i][0]
            tmp_slot_encoder_input = out_seq_train[i]

            # print (tmp_encoder_input)
            tmp_encoder_inputs.append(tmp_encoder_input)
            tmp_intent_decoder_inputs.append(tmp_intent_decoder_input)
            tmp_slot_decoder_inputs.append(tmp_slot_encoder_input)

        ## Padding
        for i in xrange(batch_size):
            encoder_pad = [PAD_ID] * (max_size - len(tmp_encoder_inputs[i]))
            # print(tmp_encoder_inputs[i])
            encoder_inputs.append(tmp_encoder_inputs[i] + encoder_pad)

            # Do NOT Pad the Intent decoder for this MODEL
            # decoder_pad_size = max_size - len(tmp_intent_decoder_inputs[i])
            # intent_decoder_inputs.append( tmp_intent_decoder_inputs[i] + tmp_intent_decoder_inputs[i] * decoder_pad_size)
            intent_decoder_inputs.append(tmp_intent_decoder_inputs[i])

            decoder_pad_size = [PAD_ID] * (max_size - len(tmp_slot_decoder_inputs[i]))
            slot_decoder_inputs.append(tmp_slot_decoder_inputs[i] + decoder_pad_size)
        # Numpy Array Formatting
        batch_encoder_inputs, batch_intent_decoder_inputs, batch_slot_decoder_inputs, \
        batch_slots_weights, batch_encoder_inputs_rev = [], [], [], [], []

        for length_idx in xrange(max_size):
            batch_encoder_inputs.append(
                np.array([encoder_inputs[batch_idx][length_idx]
                          for batch_idx in xrange(batch_size)], dtype=np.int32))

            batch_slot_decoder_inputs.append(
                np.array([slot_decoder_inputs[batch_idx][length_idx]
                          for batch_idx in xrange(batch_size)], dtype=np.int32))

            batch_slot_weight = np.ones(batch_size, dtype=np.float32)
            for batch_idx in xrange(batch_size):
                if length_idx < max_size - 1:
                    target = slot_decoder_inputs[batch_idx][length_idx + 1]
                if target == PAD_ID:
                    batch_slot_weight[batch_idx] = 0
            batch_slots_weights.append(batch_slot_weight)

        for length_idx in range(max_size - 1, -1, -1):
            batch_encoder_inputs_rev.append(
                np.array([encoder_inputs[batch_idx][length_idx]
                          for batch_idx in xrange(batch_size)], dtype=np.int32))

        batch_intent_decoder_inputs.append(
            np.array([intent_decoder_inputs[batch_idx] for batch_idx in xrange(batch_size)], dtype=np.int32))

        # UPDATE in_seq_train, out_seq_train, label_train by removing the indices of this batch from the remaining
        # in_seq_train_new = [in_seq_train[i] for i in range(0, len(in_seq_train)) if i not in rnd_indices]
        # out_seq_train_new = [out_seq_train[i] for i in range(0, len(out_seq_train)) if i not in rnd_indices]
        # label_train_new = [label_train[i] for i in range(0, len(label_train)) if i not in rnd_indices]

        return batch_encoder_inputs, batch_encoder_inputs_rev, batch_intent_decoder_inputs, batch_slot_decoder_inputs, \
               batch_slots_weights, max_size  # , in_seq_train_new, out_seq_train_new, label_train_new
