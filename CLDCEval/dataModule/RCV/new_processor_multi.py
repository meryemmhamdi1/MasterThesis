from keras.utils import to_categorical
import csv
import os
import cPickle as pkl

SEPARATOR = "\t"

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

        train_name = self.data_path + "/train.csv"
        dev_name = self.data_path + "/dev.csv"
        test_name = self.data_path + "/test.csv"

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

        return x_train, one_hot_train, x_dev, one_hot_dev, x_test, one_hot_test

