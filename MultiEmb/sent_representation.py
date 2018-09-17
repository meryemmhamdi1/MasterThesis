from string import punctuation

import nltk
import numpy as np
from nltk.corpus import stopwords
"""
Sentence Representation using different model variations:
    * Sent2Vec
    * Average of words
    * Bi-directional LSTM + CNN
"""
def num_there(s):
    return not any(i.isdigit() for i in s)

def all_punct(s):
    return not all(i in punctuation for i in s)

class SentRepresentation(object):
    def __init__(self, sents_de, sents_en, vectors_de, vectors_en, method):
        self.sents_de = sents_de
        self.sents_en = sents_en
        self.vectors_de = vectors_de
        self.vectors_en = vectors_en
        self.method = method

        if self.method == "word_avg":
            tokens_de = self.tokenize(self.sents_de, "german")
            tokens_en = self.tokenize(self.sents_en, "english")
            self.sents_rep_de  = self.word_average(tokens_de, vectors_de)
            self.sents_rep_en = self.word_average(tokens_en, vectors_en)

        elif self.method == "sent2vec":
            sent2vec_de = self.load_sent2Vec("german")
            sent2vec_en = self.load_sent2Vec("english")

            self.sents_rep_de = self.apply_sent2Vec(self.sents_de, sent2vec_de)
            self.sents_rep_en = self.apply_sent2Vec(self.sents_en, sent2vec_en)

    def tokenize_stop_word(self, sents, language):
        tokens_list = []  # List of list of tokens
        for i in tqdm(range(0, len(sents))):
            ## Tokenizing each sentence in doc
            tokens = nltk.word_tokenize(sents[i])
            tokens_sent = [word.lower() for word in tokens if word not in punctuation and
                           word not in stopwords.words(language)]
            tokens_list.append(tokens_sent)

        return tokens_list

    def word_average(self, tokens, vectors):
        sents_vec = []
        for sent in tokens:
            words_vec = []
            for token in sent:
                words_vec.append(vectors[token])
            sents_vec.append(np.mean(words_vec))
        return sents_vec

    """
    def load_sent2Vec(self, language):

    def apply_sent2Vec(self):
    def bi_lstm_att(self):
        ## Learns the parameters of bi_lstm by optimizing the similarity over the set of aligned sentences


    def joint_word_sent_opt(self):
        ## Using solution of Mikolov using jointly training and optimizing using sgd or ada weighted sum of
        # the squared reconstruction error of word transformation and sentence transformation
        
    """


