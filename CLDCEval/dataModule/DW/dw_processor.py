import os
import cPickle as pkl
import numpy as np


class DWProcessor(object):

    def __init__(self, data_util, model_name, mode):
        ## 1. Create a dictionary which maps from word ids to words using word2vec vocabularies and save it for later reuse
        ##    and Apply conversion from word ids to actual words
        # Train
        print ("Creating a dictionary which maps from word ids to words using word2vec vocabularies =>")
        print ("For Train ...")
        x_train, y_train = data_util.load_w2v_model_map_ids("train")
        print("len(x_train)=",len(x_train))
        #print(x_train[0])
        print("len(y_train)=",len(y_train))
        #print(y_train[0])

        # Dev
        print ("For Dev ...")
        x_dev, y_dev  = data_util.load_w2v_model_map_ids("dev")
        print("len(x_dev)=",len(x_dev))
        #print(x_dev[0])
        print("len(y_dev)=",len(y_dev))
        #print(y_dev[0])

        # Test
        print ("For Test ...")
        x_test, y_test  = data_util.load_w2v_model_map_ids("test")
        print("len(x_test)=",len(x_test))
        #print(x_test[0])
        print("len(y_test)=",len(y_test))
        #print(y_test[0])


        ## 2. Apply the multilingual model to convert words to multilingual vectors
        print ("Apply the multilingual model to convert words to multilingual vectors")
        model_path = data_util.emb_model_path + data_util.emb_model_name + "/" + data_util.language+"_vector_model.p"
        if not os.path.isfile(model_path):
            data_util.load_multi_vectors()

        print ("Loading word2vec model for the language")
        with open(data_util.emb_model_path + data_util.emb_model_name + "/" + data_util.language+"_vector_model.p", "rb") \
                as model_file:
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

        ## 3. Create Vocabulary out of the target labels and Look for the significance of the labels #
        # Combining train, dev and test datasets to construct common vocabulary for each language
        print ("Creating Vocabulary out of the target labels")
        y = y_train + y_dev + y_test
        dict_map_path = data_util.pre_dir+data_util.language+"_mapping_labels.p"
        if not os.path.isfile(dict_map_path):
            dict_map = data_util.create_id_mapping(y, dict_map_path)

            # label_vocab_path = os.path.join(data_util.pre_dir, data_util.language + "label_vocab.txt")
            # label_vocab_list, label_vocab = data_util.create_vocab(label_vocab_path, y)

            ## Convert target labels to ids for now
            ## Upload the mapping dictionary
        else:
            print ("Loading mapping dictionary")
            with open(dict_map_path,"rb") as dict_pkl:
                dict_map = pkl.load(dict_pkl)

        print ("Converting target labels to ids")
        label_ids_train = data_util.label_to_ids_1(y_train,  dict_map)
        label_ids_dev = data_util.label_to_ids_1(y_dev, dict_map)
        label_ids_test = data_util.label_to_ids_1(y_test, dict_map)

        self.x_vec_train = x_vec_train
        self.label_ids_train = label_ids_train
        self.x_vec_dev = x_vec_dev
        self.label_ids_dev = label_ids_dev
        self.x_vec_test = x_vec_test
        self.label_ids_test = label_ids_test

