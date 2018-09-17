from keras.utils import to_categorical
import pandas as pd
from tqdm import tqdm
import csv
import numpy as np
import os
import ast
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

            print("6 Removing empty and very long documents and their corresponding labels")
            x_vec_train_new = []
            label_ids_train_new = []
            for i in range(0, len(x_train_new)):
                if 0 < len(x_train_new[i]) < MAX_SEQUENCES:
                    x_vec_train_new.append(x_train_new[i])
                    label_ids_train_new.append(self.y_train[i])

            x_vec_dev_new = []
            label_ids_dev_new = []
            for i in range(0, len(x_dev_new)):
                if 0 < len(x_dev_new[i]) < MAX_SEQUENCES:
                    x_vec_dev_new.append(x_dev_new[i])
                    label_ids_dev_new.append(self.y_dev[i])

            x_vec_test_new = []
            label_ids_test_new = []
            for i in range(0, len(x_test_new)):
                if 0 < len(x_test_new[i]) < MAX_SEQUENCES:
                    x_vec_test_new.append(x_test_new[i])
                    label_ids_test_new.append(self.y_test[i])

            self.x_train_pro = x_vec_train_new
            self.x_dev_pro = x_vec_dev_new
            self.x_test_pro = x_vec_test_new
            self.y_train = label_ids_train_new
            self.y_dev = label_ids_dev_new
            self.y_test = label_ids_test_new

    """ Reading of data from Text files"""
    def load_data(self):
        # Load fine grained categories
        path_categories = "../dataModule/RCV/RCVCategories.txt"
        categories = {}
        i = 0
        with open(path_categories, "r") as categories_file:
            for line in categories_file:
                cat = line.split(" ")[0]
                categories.update({cat: i})
                i += 1

        # Load x_train, y_train, x_dev, y_dev, x_test, y_test
        train_name = self.data_path + "/train.csv"
        dev_name = self.data_path + "/dev.csv"
        test_name = self.data_path + "/test.csv"

        # Train Data
        train_df = pd.read_csv(train_name)
        x_train = list(train_df["texts"])
        x_train_join = []
        for i in tqdm(range(0, len(x_train))):
            x_train_join.append(" ".join(x_train[i][:-1]))

        y_train = []
        for i in tqdm(range(0, len(train_df))):
            y_labels_sub = []
            for cat in categories.keys():
                y_labels_sub.append(train_df.iloc[i][cat])
            y_train.append(y_labels_sub)

        # Dev Data
        dev_df = pd.read_csv(dev_name)
        x_dev = list(dev_df["texts"])
        x_dev_join = []
        for i in tqdm(range(0, len(x_dev))):
            x_dev_join.append(" ".join(x_dev[i][:-1]))

        y_dev = []
        for i in tqdm(range(0, len(dev_df))):
            y_labels_sub = []
            for cat in categories.keys():
                y_labels_sub.append(dev_df.iloc[i][cat])
            y_dev.append(y_labels_sub)

        # Test Data
        test_df = pd.read_csv(test_name)
        x_test = list(test_df["texts"])
        x_test_join = []
        for i in tqdm(range(0, len(x_test))):
            x_test_join.append(" ".join(x_test[i][:-1]))

        y_test = []
        for i in tqdm(range(0, len(test_df))):
            y_labels_sub = []
            for cat in categories.keys():
                y_labels_sub.append(test_df.iloc[i][cat])
            y_test.append(y_labels_sub)

        self.n_classes = len(categories.keys())
        print("n_classes=", self.n_classes)

        return x_train, y_train, x_dev, y_dev, x_test, y_test

class RCVProcessorFineGrained(object):

    def __init__(self, data_util, lang_dict, n_classes, single_label):
        self.data_util = data_util
        self.data_path = self.data_util.data_root + self.data_util.language
        self.n_classes = n_classes
        self.single_label = single_label

        print("2. Data Preprocessing ...")
        if self.single_label:
            x_train_pro_p = self.data_path + "/x_train_pro_single_label.p"
            x_dev_pro_p = self.data_path + "/x_dev_pro_single_label.p"
            x_test_pro_p = self.data_path + "/x_test_pro_single_label.p"

            if self.n_classes == 4:
                y_train_p = self.data_path + "/y_train_single_4_label.p"
                y_dev_p = self.data_path + "/y_dev_single_4_label.p"
                y_test_p = self.data_path + "/y_test_single_4_label.p"

            else:
                self.n_classes = 55
                y_train_p = self.data_path + "/y_train_single_55_label.p"
                y_dev_p = self.data_path + "/y_dev_single_55_label.p"
                y_test_p = self.data_path + "/y_test_single_55_label.p"

        else:
            x_train_pro_p = self.data_path + "/x_train_pro_multi_label.p"
            x_dev_pro_p = self.data_path + "/x_dev_pro_multi_label.p"
            x_test_pro_p = self.data_path + "/x_test_pro_multi_label.p"

            y_train_p = self.data_path + "/y_train_multi_label.p"
            y_dev_p = self.data_path + "/y_dev_multi_label.p"
            y_test_p = self.data_path + "/y_test_multi_label.p"
            self.n_classes = 103

        if not os.path.isfile(x_train_pro_p) or not os.path.isfile(x_dev_pro_p) or not os.path.isfile(x_test_pro_p)\
                or not os.path.isfile(y_train_p) or not os.path.isfile(y_dev_p) or not os.path.isfile(y_test_p):
            print("Loading Data")
            if single_label:
                if self.n_classes == 4:
                    x_train, y_train, x_dev, y_dev, x_test, y_test = self.load_data_4_single_labels()
                else:
                    x_train, y_train, x_dev, y_dev, x_test, y_test = self.load_data_55_single_labels()
            else:
                x_train, y_train, x_dev, y_dev, x_test, y_test = self.load_data_multi_labels()
                self.n_classes = 103

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
                self.x_train_pro = pkl.load(file)

            with open(x_dev_pro_p, "rb") as file:
                self.x_dev_pro = pkl.load(file)

            with open(x_test_pro_p, "rb") as file:
                self.x_test_pro = pkl.load(file)

            print("Loading from Pickle files ...")
            with open(y_train_p, "rb") as file:
                self.y_train = np.array(pkl.load(file))

            with open(y_dev_p, "rb") as file:
                self.y_dev = np.array(pkl.load(file))

            with open(y_test_p, "rb") as file:
                self.y_test = np.array(pkl.load(file))
                
        """
        print("Loading from Pickle files ...")
        with open(x_train_pro_p, "rb") as file:
            self.x_train_pro = pkl.load(file)

        with open(x_dev_pro_p, "rb") as file:
            self.x_dev_pro = pkl.load(file)

        with open(x_test_pro_p, "rb") as file:
            self.x_test_pro = pkl.load(file)

        print("Loading from Pickle files ...")
        with open(y_train_p, "rb") as file:
            y_train = np.array(pkl.load(file)).argmax(1)

        with open(y_dev_p, "rb") as file:
            y_dev = np.array(pkl.load(file)).argmax(1)

        with open(y_test_p, "rb") as file:
            y_test = np.array(pkl.load(file)).argmax(1)

        path_categories = "../dataModule/RCV/55_categories.txt"
        categories = {}
        i = 0
        with open(path_categories, "r") as categories_file:
            for line in categories_file:
                categories.update({i: line.rstrip("\n")})
                i += 1

        y_train_4 = []
        for y in y_train:
            if categories[y][0] == "C":
                y_train_4.append(0)
            elif categories[y][0] == "E":
                y_train_4.append(1)
            elif categories[y][0] == "G":
                y_train_4.append(2)
            elif categories[y][0] == "M":
                y_train_4.append(3)

        y_dev_4 = []
        for y in y_dev:
            if categories[y][0] == "C":
                y_dev_4.append(0)
            elif categories[y][0] == "E":
                y_dev_4.append(1)
            elif categories[y][0] == "G":
                y_dev_4.append(2)
            elif categories[y][0] == "M":
                y_dev_4.append(3)

        y_test_4 = []
        for y in y_test:
            if categories[y][0] == "C":
                y_test_4.append(0)
            elif categories[y][0] == "E":
                y_test_4.append(1)
            elif categories[y][0] == "G":
                y_test_4.append(2)
            elif categories[y][0] == "M":
                y_test_4.append(3)

        self.y_train = to_categorical(y_train_4, num_classes=self.n_classes)
        self.y_dev = to_categorical(y_dev_4, num_classes=self.n_classes)
        self.y_test = to_categorical(y_test_4, num_classes=self.n_classes)

        print("Saving to Pickle files ...")
        with open(y_train_s_p, "wb") as file:
            pkl.dump(self.y_train, file)

        with open(y_dev_s_p, "wb") as file:
            pkl.dump(self.y_dev, file)

        with open(y_test_s_p, "wb") as file:
            pkl.dump(self.y_test, file)
            
        """


    """ Reading of data from Text files"""
    def load_data_single_labels(self):
        # Load 55 grained categories
        path_categories = "../dataModule/RCV/55_categories.txt"
        categories = {}
        i = 0
        with open(path_categories, "r") as categories_file:
            for line in categories_file:
                categories.update({line.rstrip("\n"): i})
                i += 1

        c_categories = []
        e_categories = []
        g_categories = []
        m_categories = []
        for cate in categories:
            if cate[0] == "C":
                c_categories.append(cate)
            elif cate[0] == "E":
                e_categories.append(cate)
            elif cate[0] == "G":
                g_categories.append(cate)
            elif cate[0] == "M":
                m_categories.append(cate)

        train_name = self.data_path + "/train.csv"
        dev_name = self.data_path + "/dev.csv"
        test_name = self.data_path + "/test.csv"

        train_df = pd.read_csv(train_name).fillna("sterby")
        dev_df = pd.read_csv(dev_name).fillna("sterby")
        test_df = pd.read_csv(test_name).fillna("sterby")

        y_train_df = train_df[train_df["lead_topic"] != "NONE"]

        label_dict = {'CCAT': 0, 'ECAT': 1, 'GCAT': 2, 'MCAT': 3}
        y_train_list = []
        y_train_4_list = []
        i_indices_train = []
        print("Loading y_train ...")
        for i in tqdm(range(0, len(y_train_df))):
            if y_train_df.iloc[i]["lead_topic"] == "CCAT":
                for cate in c_categories:
                    if y_train_df.iloc[i][cate] == 1:
                        y_train_list.append(categories[cate])
                        y_train_4_list.append(label_dict["CCAT"])
                        i_indices_train.append(i)
                        break
            elif y_train_df.iloc[i]["lead_topic"] == "ECAT":
                for cate in e_categories:
                    if y_train_df.iloc[i][cate] == 1:
                        y_train_list.append(categories[cate])
                        y_train_4_list.append(label_dict["ECAT"])
                        i_indices_train.append(i)
                        break
            elif y_train_df.iloc[i]["lead_topic"] == "GCAT":
                for cate in g_categories:
                    if y_train_df.iloc[i][cate] == 1:
                        y_train_list.append(categories[cate])
                        y_train_4_list.append(label_dict["GCAT"])
                        i_indices_train.append(i)
                        break

            elif y_train_df.iloc[i]["lead_topic"] == "MCAT":
                for cate in m_categories:
                    if y_train_df.iloc[i][cate] == 1:
                        y_train_list.append(categories[cate])
                        y_train_4_list.append(label_dict["MCAT"])
                        i_indices_train.append(i)
                        break

        print("Loading texts_train ...")
        texts_train = []
        for i in tqdm(range(0, len(i_indices_train))):
            texts_train.append(" ".join(ast.literal_eval(y_train_df.iloc[i]["texts"])[1:]))

        ###
        y_dev_df = dev_df[dev_df["lead_topic"] != "NONE"]

        y_dev_list = []
        y_dev_4_list = []
        i_indices_dev = []
        print("Loading y_dev ...")
        for i in tqdm(range(0, len(y_dev_df))):
            if y_dev_df.iloc[i]["lead_topic"] == "CCAT":
                for cate in c_categories:
                    if y_dev_df.iloc[i][cate] == 1:
                        y_dev_list.append(categories[cate])
                        y_dev_4_list.append(label_dict["CCAT"])
                        i_indices_dev.append(i)
                        break
            elif y_dev_df.iloc[i]["lead_topic"] == "ECAT":
                for cate in e_categories:
                    if y_dev_df.iloc[i][cate] == 1:
                        y_dev_list.append(categories[cate])
                        y_dev_4_list.append(label_dict["ECAT"])
                        i_indices_dev.append(i)
                        break
            elif y_dev_df.iloc[i]["lead_topic"] == "GCAT":
                for cate in g_categories:
                    if y_dev_df.iloc[i][cate] == 1:
                        y_dev_list.append(categories[cate])
                        y_dev_4_list.append(label_dict["GCAT"])
                        i_indices_dev.append(i)
                        break

            elif y_dev_df.iloc[i]["lead_topic"] == "MCAT":
                for cate in m_categories:
                    if y_dev_df.iloc[i][cate] == 1:
                        y_dev_list.append(categories[cate])
                        y_dev_4_list.append(label_dict["MCAT"])
                        i_indices_dev.append(i)
                        break

        print("Loading texts_dev ...")
        texts_dev = []
        for i in tqdm(range(0, len(i_indices_dev))):
            texts_dev.append(" ".join(ast.literal_eval(y_dev_df.iloc[i]["texts"])[1:]))

        ####
        y_test_df = test_df[test_df["lead_topic"] != "NONE"]

        y_test_list = []
        y_test_4_list = []
        i_indices_test = []
        print("Loading y_test ...")
        for i in tqdm(range(0, len(y_test_df))):
            if y_test_df.iloc[i]["lead_topic"] == "CCAT":
                for cate in c_categories:
                    if y_test_df.iloc[i][cate] == 1:
                        y_test_list.append(categories[cate])
                        y_test_4_list.append(label_dict["CCAT"])
                        i_indices_test.append(i)
                        break
            elif y_test_df.iloc[i]["lead_topic"] == "ECAT":
                for cate in e_categories:
                    if y_test_df.iloc[i][cate] == 1:
                        y_test_list.append(categories[cate])
                        y_test_4_list.append(label_dict["ECAT"])
                        i_indices_test.append(i)
                        break
            elif y_test_df.iloc[i]["lead_topic"] == "GCAT":
                for cate in g_categories:
                    if y_test_df.iloc[i][cate] == 1:
                        y_test_list.append(categories[cate])
                        y_test_4_list.append(label_dict["GCAT"])
                        i_indices_test.append(i)
                        break

            elif y_test_df.iloc[i]["lead_topic"] == "MCAT":
                for cate in m_categories:
                    if y_test_df.iloc[i][cate] == 1:
                        y_test_list.append(categories[cate])
                        y_test_4_list.append(label_dict["MCAT"])
                        i_indices_test.append(i)
                        break

        print("Loading texts_test ...")
        texts_test = []
        for i in tqdm(range(0, len(i_indices_test))):
            texts_test.append(" ".join(ast.literal_eval(y_test_df.iloc[i]["texts"])[1:]))

        return texts_train, y_train_list, y_train_4_list, texts_dev, y_dev_list, \
               y_dev_4_list, texts_test, y_test_list, y_test_4_list

    def load_data_4_single_labels(self):

        texts_train, y_train_list, y_train_4_list, texts_dev, y_dev_list, \
        y_dev_4_list, texts_test, y_test_list, y_test_4_list = self.load_data_single_labels()


        self.n_classes = 4
        print("n_classes=", self.n_classes)

        print("One hot encoding")
        one_hot_train = to_categorical(y_train_4_list, num_classes=self.n_classes)
        one_hot_dev = to_categorical(y_dev_4_list, num_classes=self.n_classes)
        one_hot_test = to_categorical(y_test_4_list, num_classes=self.n_classes)

        return texts_train, one_hot_train, texts_dev, one_hot_dev, texts_test, one_hot_test


    def load_data_55_single_labels(self):
        texts_train, y_train_list, y_train_4_list, texts_dev, y_dev_list, \
        y_dev_4_list, texts_test, y_test_list, y_test_4_list = self.load_data_single_labels()

        self.n_classes = 55

        print("One hot encoding")
        one_hot_train = to_categorical(y_train_list, num_classes=self.n_classes)
        one_hot_dev = to_categorical(y_dev_list, num_classes=self.n_classes)
        one_hot_test = to_categorical(y_test_list, num_classes=self.n_classes)

        print("n_classes=", self.n_classes)

        return texts_train, one_hot_train, texts_dev, one_hot_dev, texts_test, one_hot_test

    def load_data_multi_labels(self):
        # Load fine grained categories
        path_categories = "../dataModule/RCV/RCVCategories.txt"
        categories = {}
        i = 0
        with open(path_categories, "r") as categories_file:
            for line in categories_file:
                cat = line.split(" ")[0]
                categories.update({cat: i})
                i += 1

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

        one_hot_train = train_df[list(categories.keys())].values
        one_hot_dev = dev_df[list(categories.keys())].values
        one_hot_test = test_df[list(categories.keys())].values

        self.n_classes = len(one_hot_train[0])
        print("n_classes=", self.n_classes)

        return X_train, one_hot_train, X_dev, one_hot_dev, X_test, one_hot_test


class RCVProcessor(object):

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

            print("Removing empty and very long documents and their corresponding labels")

            x_vec_train_new = []
            label_ids_train_new = []
            for i in range(0, len(self.x_train_pro)):
                if 0 < len(self.x_train_pro[i]) < MAX_SEQUENCES:
                    x_vec_train_new.append(self.x_train_pro[i])
                    label_ids_train_new.append(self.y_train[i])

            x_vec_dev_new = []
            label_ids_dev_new = []
            for i in range(0, len(self.x_dev_pro)):
                if 0 < len(self.x_dev_pro[i]) < MAX_SEQUENCES:
                    x_vec_dev_new.append(self.x_dev_pro[i])
                    label_ids_dev_new.append(self.y_dev[i])

            x_vec_test_new = []
            label_ids_test_new = []
            for i in range(0, len(self.x_test_pro)):
                if 0 < len(self.x_test_pro[i]) < MAX_SEQUENCES:
                    x_vec_test_new.append(self.x_test_pro[i])
                    label_ids_test_new.append(self.y_test[i])

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

            self.x_train_pro = x_train_new
            self.x_dev_pro = x_dev_new
            self.x_test_pro = x_test_new
            self.y_train = label_ids_train_new
            self.y_dev = label_ids_dev_new
            self.y_test = label_ids_test_new

    """ Reading of data from Text files"""
    def load_data(self):
        x_train = []
        y_train = []
        y_train_55 = []
        x_dev = []
        y_dev = []
        y_dev_55 = []
        x_test = []
        y_test = []
        y_test_55 = []

        path_categories = "../dataModule/RCV/55_categories.txt"
        categories = {}
        i = 0
        with open(path_categories, "r") as categories_file:
            for line in categories_file:
                categories.update({line.rstrip("\n"): i})
                i += 1

        c_categories = []
        e_categories = []
        g_categories = []
        m_categories = []
        for cate in categories:
            if cate[0] == "C":
                c_categories.append(cate)
            elif cate[0] == "E":
                e_categories.append(cate)
            elif cate[0] == "G":
                g_categories.append(cate)
            elif cate[0] == "M":
                m_categories.append(cate)

        train_name = self.data_path + "/train.csv"
        dev_name = self.data_path + "/dev.csv"
        test_name = self.data_path + "/test.csv"

        label_dict = {'CCAT': 0, 'ECAT': 1, 'GCAT': 2, 'MCAT': 3}

        train_df = pd.read_csv(train_name)
        train_df = train_df[train_df["lead_topic"] != "NONE"].reset_index()
        for i in tqdm(range(0, len(train_df))):
            y_train_55_labels = []
            if train_df.iloc[i]["lead_topic"] == "CCAT":
                for cate in c_categories:
                    if train_df.iloc[i][cate] == 1:
                        y_train_55_labels.append(categories[cate])
            elif train_df.iloc[i]["lead_topic"] == "GCAT":
                for cate in g_categories:
                    if train_df.iloc[i][cate] == 1:
                        y_train_55_labels.append(categories[cate])

            elif train_df.iloc[i]["lead_topic"] == "ECAT":
                for cate in e_categories:
                    if train_df.iloc[i][cate] == 1:
                        y_train_55_labels.append(categories[cate])

            else:
                for cate in m_categories:
                    if train_df.iloc[i][cate] == 1:
                        y_train_55_labels.append(categories[cate])

            if len(y_train_55_labels) > 0:
                y_train_55.append(y_train_55_labels[0])
                x_train.append(" ".join(ast.literal_eval(train_df.iloc[i]["texts"])))
                y_train.append(label_dict[train_df.iloc[i]["lead_topic"]])

        dev_df = pd.read_csv(dev_name)
        dev_df = dev_df[dev_df["lead_topic"] != "NONE"].reset_index()
        for i in tqdm(range(0, len(dev_df))):
            y_dev_55_labels = []
            if dev_df.iloc[i]["lead_topic"] == "CCAT":
                for cate in c_categories:
                    if dev_df.iloc[i][cate] == 1:
                        y_dev_55_labels.append(categories[cate])
            elif dev_df.iloc[i]["lead_topic"] == "GCAT":
                for cate in g_categories:
                    if dev_df.iloc[i][cate] == 1:
                        y_dev_55_labels.append(categories[cate])

            elif dev_df.iloc[i]["lead_topic"] == "ECAT":
                for cate in e_categories:
                    if dev_df.iloc[i][cate] == 1:
                        y_dev_55_labels.append(categories[cate])

            else:
                for cate in m_categories:
                    if dev_df.iloc[i][cate] == 1:
                        y_dev_55_labels.append(categories[cate])

            if len(y_dev_55_labels) > 0:
                y_dev_55.append(y_dev_55_labels[0])
                x_dev.append(" ".join(ast.literal_eval(dev_df.iloc[i]["texts"])))
                y_dev.append(label_dict[dev_df.iloc[i]["lead_topic"]])

        test_df = pd.read_csv(test_name)
        test_df = test_df[test_df["lead_topic"] != "NONE"].reset_index()
        for i in tqdm(range(0, len(test_df))):
            y_test_55_labels = []
            if test_df.iloc[i]["lead_topic"] == "CCAT":
                for cate in c_categories:
                    if test_df.iloc[i][cate] == 1:
                        y_test_55_labels.append(categories[cate])
            elif test_df.iloc[i]["lead_topic"] == "GCAT":
                for cate in g_categories:
                    if test_df.iloc[i][cate] == 1:
                        y_test_55_labels.append(categories[cate])

            elif test_df.iloc[i]["lead_topic"] == "ECAT":
                for cate in e_categories:
                    if test_df.iloc[i][cate] == 1:
                        y_test_55_labels.append(categories[cate])

            else:
                for cate in m_categories:
                    if test_df.iloc[i][cate] == 1:
                        y_test_55_labels.append(categories[cate])

            if len(y_test_55_labels) > 0:
                y_test_55.append(y_test_55_labels[0])
                x_test.append(" ".join(ast.literal_eval(test_df.iloc[i]["texts"])))
                y_test.append(label_dict[test_df.iloc[i]["lead_topic"]])

        """
            
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
                
        """

        self.n_classes = 55
        print("n_classes=", self.n_classes)

        print("One hot encoding")
        one_hot_train = to_categorical(list(y_train_55), num_classes=self.n_classes)
        one_hot_dev = to_categorical(list(y_dev_55), num_classes=self.n_classes)
        one_hot_test = to_categorical(list(y_test_55), num_classes=self.n_classes)

        return x_train, one_hot_train, x_dev, one_hot_dev, x_test, one_hot_test
