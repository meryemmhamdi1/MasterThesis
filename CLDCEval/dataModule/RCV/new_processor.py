from keras.utils import to_categorical
import csv
import os
import pandas as pd
import ast
from tqdm import tqdm
import numpy as np
try:
    import cPickle as pkl
except:
    import _pickle as pkl

SEPARATOR = "\t"
MAX_SEQUENCES = 622


class GeneralProcessor(object):

    def __init__(self, data_util, lang_dict):
        self.data_util = data_util
        self.data_path = self.data_util.data_root + self.data_util.language
        print("1. Loading data ....")
        x_train, self.y_train, x_dev, self.y_dev, x_test, self.y_test = self.load_data()

        print("2. Data Preprocessing ...")
        x_train_pro_p = self.data_path + "/x_train_pro.p"
        x_dev_pro_p = self.data_path + "/x_dev_pro.p"
        x_test_pro_p = self.data_path + "/x_test_pro.p"

        if not os.path.isfile(x_train_pro_p) or not os.path.isfile(x_dev_pro_p) or not os.path.isfile(x_test_pro_p):
            self.x_train_pro = self.data_util.nltk_tokenizer_flat(x_train)
            self.x_dev_pro = self.data_util.nltk_tokenizer_flat(x_dev)
            self.x_test_pro = self.data_util.nltk_tokenizer_flat(x_test)

            print("Saving to Pickle files ...")
            with open(x_train_pro_p, "wb") as file:
                pkl.dump(self.x_train_pro, file)

            with open(x_dev_pro_p, "wb") as file:
                pkl.dump(self.x_dev_pro, file)

            with open(x_test_pro_p, "wb") as file:
                pkl.dump(self.x_test_pro, file)
        else:
            print("Loading from Pickle files ...")
            with open(x_train_pro_p, "rb") as file:
                self.x_train_pro = pkl.load(file)

            with open(x_dev_pro_p, "rb") as file:
                self.x_dev_pro = pkl.load(file)

            with open(x_test_pro_p, "rb") as file:
                self.x_test_pro = pkl.load(file)

            print("Adding language prefix to the word")
            x_train_new = []
            for x_train in self.x_train_pro:
                x_train_new_sub = []
                for word in x_train:
                    x_train_new_sub.append(lang_dict[self.data_util.language]+"_"+word)
                x_train_new.append(x_train_new_sub)

            x_dev_new = []
            for x_dev in self.x_dev_pro:
                x_dev_new_sub = []
                for word in x_dev:
                    x_dev_new_sub.append(lang_dict[self.data_util.language]+"_"+word)
                x_dev_new.append(x_dev_new_sub)

            x_test_new = []
            for x_test in self.x_test_pro:
                x_test_new_sub = []
                for word in x_test:
                    x_test_new_sub.append(lang_dict[self.data_util.language]+"_"+word)
                x_test_new.append(x_test_new_sub)

            self.x_train_pro = x_train_new
            self.x_dev_pro = x_dev_new
            self.x_test_pro = x_test_new

    """ Reading of data from Text files"""
    def load_data(self):
        x_train = []
        y_train = []
        x_dev = []
        y_dev = []
        x_test = []
        y_test = []

        train_name = self.data_path + "/train.txt"
        dev_name = self.data_path + "/dev.txt"
        test_name = self.data_path + "/test.txt"

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

        self.n_classes = len(set(list(y_train)))
        print("n_classes=",self.n_classes)

        print("One hot encoding")
        one_hot_train = to_categorical(list(y_train), num_classes=self.n_classes)
        one_hot_dev = to_categorical(list(y_dev), num_classes=self.n_classes)
        one_hot_test = to_categorical(list(y_test), num_classes=self.n_classes)

        return x_train, one_hot_train, x_dev, one_hot_dev, x_test, one_hot_test

class RCVProcessor(object):

    def __init__(self, data_util, lang_dict):
        self.data_util = data_util
        self.data_path = self.data_util.data_root + self.data_util.language

        print("2. Data Preprocessing ...")
        x_train_pro_p = self.data_path + "/train/X_train_processed_" + self.data_util.language + ".p" #"/x_train_pro_single_label.p"
        x_dev_pro_p = self.data_path + "/dev/X_dev_processed_" + self.data_util.language + ".p" #"/x_dev_pro_single_label.p"
        x_test_pro_p = self.data_path + "/test/X_test_processed_" + self.data_util.language + ".p" #"/x_test_pro_single_label.p"

        y_train_p = self.data_path + "/train/y_train_" + self.data_util.language + ".p" #"/y_train_single_4_label.p"
        y_dev_p = self.data_path + "/dev/y_dev_" + self.data_util.language + ".p" #"/y_dev_single_4_label.p"
        y_test_p = self.data_path + "/test/y_test_" + self.data_util.language + ".p" #"/y_test_single_4_label.p"

        if not os.path.isfile(x_train_pro_p) or not os.path.isfile(x_dev_pro_p) or not os.path.isfile(x_test_pro_p):
            print("Loading Data")
            x_train, y_train, x_dev, y_dev, x_test, y_test = self.load_data()

            x_train_pro = self.data_util.nltk_tokenizer_flat(x_train)
            x_dev_pro = self.data_util.nltk_tokenizer_flat(x_dev)
            x_test_pro = self.data_util.nltk_tokenizer_flat(x_test)

            print("Removing empty and very long documents and their corresponding labels")

            x_vec_train_new = []
            label_ids_train_new = []
            for i in range(0, len(x_train_pro)):
                if 0 < len(x_train_pro[i]) < MAX_SEQUENCES:
                    x_vec_train_new.append(x_train_pro[i])
                    label_ids_train_new.append(y_train[i])

            x_vec_dev_new = []
            label_ids_dev_new = []
            for i in range(0, len(x_dev_pro)):
                if 0 < len(x_dev_pro[i]) < MAX_SEQUENCES:
                    x_vec_dev_new.append(x_dev_pro[i])
                    label_ids_dev_new.append(y_dev[i])

            x_vec_test_new = []
            label_ids_test_new = []
            for i in range(0, len(x_test_pro)):
                if 0 < len(x_test_pro[i]) < MAX_SEQUENCES:
                    x_vec_test_new.append(x_test_pro[i])
                    label_ids_test_new.append(y_test[i])

            print("Adding language prefix to the word")
            x_train_new = []
            for x_train in x_vec_train_new:
                x_train_new_sub = []
                for word in x_train:
                    x_train_new_sub.append(lang_dict[self.data_util.language]+"_"+word)
                x_train_new.append(x_train_new_sub)

            x_dev_new = []
            for x_dev in x_vec_dev_new:
                x_dev_new_sub = []
                for word in x_dev:
                    x_dev_new_sub.append(lang_dict[self.data_util.language]+"_"+word)
                x_dev_new.append(x_dev_new_sub)

            x_test_new = []
            for x_test in x_vec_test_new:
                x_test_new_sub = []
                for word in x_test:
                    x_test_new_sub.append(lang_dict[self.data_util.language]+"_"+word)
                x_test_new.append(x_test_new_sub)

            print("Saving to Pickle files ...")
            with open(x_train_pro_p, "wb") as file:
                pkl.dump(x_train_new, file)

            with open(x_dev_pro_p, "wb") as file:
                pkl.dump(x_dev_new, file)

            with open(x_test_pro_p, "wb") as file:
                pkl.dump(x_test_new, file)

            print("Saving to Pickle files ...")
            with open(y_train_p, "wb") as file:
                pkl.dump(label_ids_train_new, file)

            with open(y_dev_p, "wb") as file:
                pkl.dump(label_ids_dev_new, file)

            with open(y_test_p, "wb") as file:
                pkl.dump(label_ids_test_new, file)

            self.x_train_pro = x_train_new
            self.x_dev_pro = x_dev_new
            self.x_test_pro = x_test_new
            self.y_train = np.array(label_ids_train_new)
            self.y_dev = np.array(label_ids_dev_new)
            self.y_test = np.array(label_ids_test_new)
        else:
            print("Loading from Pickle files ...")
            with open(x_train_pro_p, "rb") as file:
                x_train_pro = pkl.load(file)
                self.x_train_pro = []
                for i in range(len(x_train_pro)):
                    if len(x_train_pro[i]) > 0:
                        self.x_train_pro.append(reduce(lambda x, y: x+y, x_train_pro[i]))
                    else:
                        self.x_train_pro.append([])

            with open(x_dev_pro_p, "rb") as file:
                x_dev_pro = pkl.load(file)
                self.x_dev_pro = []
                for i in range(len(x_dev_pro)):
                    if len(x_dev_pro[i]) > 0:
                        self.x_dev_pro.append(reduce(lambda x, y: x+y, x_dev_pro[i]))
                    else:
                        self.x_dev_pro.append([])

            with open(x_test_pro_p, "rb") as file:
                x_test_pro = pkl.load(file)
                self.x_test_pro = []
                for i in range(len(x_test_pro)):
                    if len(x_test_pro[i]) > 0:
                        self.x_test_pro.append(reduce(lambda x, y: x+y, x_test_pro[i]))
                    else:
                        self.x_test_pro.append([])

            print("Loading from Pickle files ...")
            with open(y_train_p, "rb") as file:
                self.y_train = np.array(pkl.load(file))

            with open(y_dev_p, "rb") as file:
                self.y_dev = np.array(pkl.load(file))

            with open(y_test_p, "rb") as file:
                self.y_test = np.array(pkl.load(file))

            self.n_classes = len(self.y_train[0])

    """ Reading of data from Text files"""
    def load_data(self):

        train_name = self.data_path + "/train.csv"
        dev_name = self.data_path + "/dev.csv"
        test_name = self.data_path + "/test.csv"

        train_df = pd.read_csv(train_name).fillna("sterby")
        dev_df = pd.read_csv(dev_name).fillna("sterby")
        test_df = pd.read_csv(test_name).fillna("sterby")

        texts_train = list(train_df[train_df["lead_topic"] != "NONE"]["texts"])
        X_train = []
        for x in tqdm(texts_train):
            X_train.append(" ".join(ast.literal_eval(x)[1:]))

        texts_dev = list(dev_df[dev_df["lead_topic"] != "NONE"]["texts"])
        X_dev = []
        for x in tqdm(texts_dev):
            X_dev.append(" ".join(ast.literal_eval(x)[1:]))

        texts_test = list(test_df[test_df["lead_topic"] != "NONE"]["texts"])
        X_test = []
        for x in tqdm(texts_test):
            X_test.append(" ".join(ast.literal_eval(x)[1:]))

        y_train_list = train_df[train_df["lead_topic"] != "NONE"]["lead_topic"].values
        y_dev_list = dev_df[dev_df["lead_topic"] != "NONE"]["lead_topic"].values
        y_test_list = test_df[test_df["lead_topic"] != "NONE"]["lead_topic"].values

        label_dict = {'CCAT': 0, 'ECAT': 1, 'GCAT': 2, 'MCAT': 3}
        y_train_ids = []
        for y_train in y_train_list:
            y_train_ids.append(label_dict[y_train])

        y_dev_ids = []
        for y_dev in y_dev_list:
            y_dev_ids.append(label_dict[y_dev])

        y_test_ids = []
        for y_test in y_test_list:
            y_test_ids.append(label_dict[y_test])


        self.n_classes = len(set(y_train_ids))
        print("n_classes=", self.n_classes)

        print("One hot encoding")
        one_hot_train = to_categorical(y_train_ids, num_classes=self.n_classes)
        one_hot_dev = to_categorical(y_dev_ids, num_classes=self.n_classes)
        one_hot_test = to_categorical(y_test_ids, num_classes=self.n_classes)

        return X_train, one_hot_train, X_dev, one_hot_dev, X_test, one_hot_test

