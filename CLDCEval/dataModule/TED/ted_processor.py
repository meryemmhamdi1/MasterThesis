import os
import cPickle as pkl
import numpy as np


class TEDProcessor(object):

    def __init__(self, data_util, mode):
        ## 1. Loading from pickle files
        # Train
        print ("Reading lists from pickle files=>")
        print ("For Train ...")
        with open(data_util.data_dir+"X_train_"+data_util.language+".p") as x_train_pkl:
            x_train = pkl.load(x_train_pkl)

        with open(data_util.data_dir+"Y_train_"+data_util.language+".p") as y_train_pkl:
            y_train = pkl.load(y_train_pkl)

        print(len(x_train))
        print(x_train[0])
        print(len(y_train))
        print(y_train[0])

        # Test
        print ("For Test ...")
        with open(data_util.data_dir+"X_test_"+data_util.language+".p") as x_test_pkl:
            x_test = pkl.load(x_test_pkl)

        with open(data_util.data_dir+"Y_test_"+data_util.language+".p") as y_test_pkl:
            y_test = pkl.load(y_test_pkl)

        print(len(x_test))
        print(x_test[0])
        print(len(y_test))
        print(y_test[0])

        ## Eliminate stop words


        # Dev
        print ("For Dev ...")
        x_dev = x_train[:len(x_test)]
        print(len(x_dev))
        print(x_dev[0])
        y_dev = y_train[:len(y_test)]
        print(len(y_dev))
        print(y_dev[0])

        #x_train = x_train[len(x_test):]
        #y_train = y_train[len(y_test):]



        ## 2. Apply the multilingual model to convert words to multilingual vectors
        print ("Apply the multilingual model to convert words to multilingual vectors")
        model_path = data_util.emb_model_path + data_util.emb_model_name + "/" + data_util.language+"_vector_model.p"
        if not os.path.isfile(model_path):
            data_util.load_multi_vectors()

        print ("Loading word2vec model for the language")
        with open(data_util.emb_model_path + data_util.emb_model_name + "/" + data_util.language+"_vector_model.p", "rb") as model_file:
            word_vector_dict = pkl.load(model_file)

        ## Apply to train split
        print ("For Train ...")
        x_vec_train = data_util.apply_emb_model(x_train, "train", word_vector_dict)

        ## Apply to dev split
        print ("For Dev ...")
        x_vec_dev = data_util.apply_emb_model(x_dev, "dev", word_vector_dict)

        ## Apply to test split
        print ("For Test ...")
        x_vec_test = data_util.apply_emb_model(x_test, "test", word_vector_dict)


        ## 5. Create Vocabulary out of the target labels and Look for the significance of the labels #
        # Combining train, dev and test datasets to construct common vocabulary for each language
        print ("Creating Vocabulary out of the target labels")

        pre_path = data_util.data_dir+data_util.pre_dir
        if not os.path.isdir(pre_path):
            os.makedirs(pre_path)

        label_vocab_path = os.path.join(pre_path, "label_vocab.txt")
        label_vocab_list, label_vocab = data_util.create_vocab(label_vocab_path, y_train)

        print ("Converting target labels to ids")
        label_ids_train = data_util.label_to_ids_simple(y_train, label_vocab)
        label_ids_dev = data_util.label_to_ids_simple(y_dev, label_vocab)
        label_ids_test = data_util.label_to_ids_simple(y_test, label_vocab)

        self.x_vec_train = x_vec_train
        self.label_ids_train = label_ids_train
        self.x_vec_dev = x_vec_dev
        self.label_ids_dev = label_ids_dev
        self.x_vec_test = x_vec_test
        self.label_ids_test = label_ids_test

