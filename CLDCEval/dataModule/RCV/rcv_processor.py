import cPickle as pkl
import glob
import os
from sklearn.model_selection import train_test_split
from xml.dom import minidom
from tqdm import tqdm
from gensim.models import KeyedVectors
from RCV_doc import *

SHUFFLE_SEED = 100099540
TRAIN_PERC = 0.6
dev_PERC = 0.2
TEST_PERC = 0.2
MAX_SEQUENCES = 622


def process_xml_rcv(file_):
    xmldoc = minidom.parse(file_)
    itemid = xmldoc.getElementsByTagName('newsitem')[0].attributes['itemid'].firstChild.data
    date = xmldoc.getElementsByTagName('newsitem')[0].attributes['date'].firstChild.data
    lang = xmldoc.getElementsByTagName('newsitem')[0].attributes['xml:lang'].firstChild.data
    # title = xmldoc.getElementsByTagName('title')[0].firstChild.data
    text = xmldoc.getElementsByTagName('text')[0].getElementsByTagName('p')

    # texts = [title]
    texts = []
    for i in range(0, len(text)):
        texts.append(text[i].firstChild.data)

    classes = ['bip:countries:1.0', 'bip:industries:1.0', 'bip:topics:1.0']

    codes = xmldoc.getElementsByTagName('metadata')[0].getElementsByTagName('codes')
    countries = []
    topics = []
    industries = []
    for i in range(0, len(codes)):
        class_ = codes[i].attributes['class'].value
        if class_ == classes[0]:  # 'bip:countries:1.0'
            codes_class = codes[i].getElementsByTagName('code')
            for j in range(0, len(codes_class)):
                countries.append(codes_class[j].attributes['code'].firstChild.data)
        elif class_ == classes[1]:  # 'bip:industries:1.0'
            codes_class = codes[i].getElementsByTagName('code')
            for j in range(0, len(codes_class)):
                industries.append(codes_class[j].attributes['code'].firstChild.data)
        else:
            codes_class = codes[i].getElementsByTagName('code')
            for j in range(0, len(codes_class)):
                topics.append(codes_class[j].attributes['code'].firstChild.data)

    if len(topics) == 0:
        lead_topic = "NONE"
    else:
        if topics[0][0] == 'C':
            lead_topic = "CCAT"
        elif topics[0][0] == 'E':
            lead_topic = "ECAT"
        elif topics[0][0] == 'G':
            lead_topic = "GCAT"
        elif topics[0][0] == 'M':
            lead_topic = "MCAT"
        else:
            lead_topic = "NONE"

    return itemid, date, lang, texts, countries, industries, topics, lead_topic


class RCVProcessor(object):
    def __init__(self, data_util, mode):
        x_train_p = data_util.pre_dir + "train/X_train_" + data_util.language + ".p"
        y_train_p = data_util.pre_dir + "train/y_train_" + data_util.language + ".p"
        x_dev_p = data_util.pre_dir + "dev/X_dev_" + data_util.language + ".p"
        y_dev_p = data_util.pre_dir + "dev/y_dev_" + data_util.language + ".p"
        x_test_p = data_util.pre_dir + "test/X_test_" + data_util.language + ".p"
        y_test_p = data_util.pre_dir + "test/y_test_" + data_util.language + ".p"

        print("x_train_p: ", x_train_p)
        print("y_train_p: ", y_train_p)
        print("x_dev_p: ", x_dev_p)
        print("y_dev_p: ", y_dev_p)
        print("x_test_p: ", x_test_p)
        print("y_test_p: ", y_test_p)

        if not os.path.isfile(x_train_p) or not os.path.isfile(y_train_p) or not os.path.isfile(x_dev_p) or \
                not os.path.isfile(y_dev_p) or not os.path.isfile(x_test_p) or not os.path.isfile(y_test_p):

            # 1. Reading xml files and gathering them into one big list of x features and their corresponding labels
            print(
                "1. Reading xml files and gathering them into one big list of x features and their corresponding labels")
            rcvdocs = []
            for name in tqdm(glob.glob(data_util.data_dir + "*")):
                directories = []
                directories.append(name.split("/")[-1])
                for directory in directories:
                    print(directory)
                    if directory not in ["codes", "dtds", "MD5SUMS", "processed", "train", "dev", "test"]:
                        ### Read all files inside the directories
                        for file_ in glob.glob(data_util.data_dir + directory + "/*"):
                            itemid, date, lang, texts, countries, industries, topics, lead_topic = process_xml_rcv(
                                file_)
                            rcvdocs.append(RCVDoc(itemid, date, lang, texts, countries, industries, topics, lead_topic))

            texts = []
            lead_topics = []
            for doc in rcvdocs:
                if doc.getLeadTopic() != "NONE":
                    texts.append(doc.getTexts())
                    lead_topics.append(doc.getLeadTopic())

            X_train, X_test, y_train, y_test = \
                train_test_split(texts, lead_topics, test_size=TEST_PERC, random_state=SHUFFLE_SEED)

            X_train, X_dev, y_train, y_dev = \
                train_test_split(X_train, y_train, test_size=dev_PERC, random_state=SHUFFLE_SEED)

            if not os.path.isdir(data_util.pre_dir + "train/"):
                os.makedirs(data_util.pre_dir + "train/")

            if not os.path.isdir(data_util.pre_dir + "dev/"):
                os.makedirs(data_util.pre_dir + "dev/")

            if not os.path.isdir(data_util.pre_dir + "test/"):
                os.makedirs(data_util.pre_dir + "test/")

            with open(x_train_p, "wb") as x_train_file:
                pkl.dump(X_train, x_train_file)

            with open(y_train_p, "wb") as y_train_file:
                pkl.dump(y_train, y_train_file)

            with open(x_dev_p, "wb") as x_dev_file:
                pkl.dump(X_dev, x_dev_file)

            with open(y_dev_p, "wb") as y_dev_file:
                pkl.dump(y_dev, y_dev_file)

            with open(x_test_p, "wb") as x_test_file:
                pkl.dump(X_test, x_test_file)

            with open(y_test_p, "wb") as y_test_file:
                pkl.dump(y_test, y_test_file)

        else:
            ## Reading pickle files
            with open(x_train_p, "rb") as x_train_file:
                X_train = pkl.load(x_train_file)

            with open(y_train_p, "rb") as y_train_file:
                y_train = pkl.load(y_train_file)

            with open(x_dev_p, "rb") as x_dev_file:
                X_dev = pkl.load(x_dev_file)

            with open(y_dev_p, "rb") as y_dev_file:
                y_dev = pkl.load(y_dev_file)

            with open(x_test_p, "rb") as x_test_file:
                X_test = pkl.load(x_test_file)

            with open(y_test_p, "rb") as y_test_file:
                y_test = pkl.load(y_test_file)

        ## 2. Apply Preprocessing: tokenization, lemmatization and stop words remodev
        print("2. Apply Preprocessing: tokenization, lemmatization and stop words remodev")
        x_train_process = data_util.pre_dir + "train/X_train_processed_" + data_util.language + ".p"
        x_dev_process = data_util.pre_dir + "dev/X_dev_processed_" + data_util.language + ".p"
        x_test_process = data_util.pre_dir + "test/X_test_processed_" + data_util.language + ".p"

        if not os.path.isfile(x_train_process) or not os.path.isfile(x_dev_process) or not os.path.isfile(
                x_test_process):
            X_train_tok = data_util.nltk_tokenizer(X_train, "train")

            X_dev_tok = data_util.nltk_tokenizer(X_dev, "dev")

            X_test_tok = data_util.nltk_tokenizer(X_test, "test")

            if data_util.use_stop:
                X_train_processed, kept_ind_train = data_util.eliminate_stop_words_punct(X_train_tok)

                X_dev_tok_processed, kept_ind_dev = data_util.eliminate_stop_words_punct(X_dev_tok)

                X_test_tok_processed, kept_ind_test = data_util.eliminate_stop_words_punct(X_test_tok)

                with open(x_train_process, "wb") as x_train_file:
                    pkl.dump(X_train_processed, x_train_file)

                with open(x_dev_process, "wb") as y_train_file:
                    pkl.dump(X_dev_tok_processed, y_train_file)

                with open(x_test_process, "wb") as x_dev_file:
                    pkl.dump(X_test_tok_processed, x_dev_file)

                    ## TODO: Finish this later
            else:
                with open(x_train_process, "wb") as x_train_file:
                    pkl.dump(X_train_tok, x_train_file)

                with open(x_dev_process, "wb") as x_dev_file:
                    pkl.dump(X_dev_tok, x_dev_file)

                with open(x_test_process, "wb") as x_test_file:
                    pkl.dump(X_test_tok, x_test_file)

        else:
            print("Loading x_train_process")
            with open(x_train_process, "rb") as x_train_file:
                X_train_tok = pkl.load(x_train_file)

            print("Loading x_dev_process")
            with open(x_dev_process, "rb") as x_dev_file:
                X_dev_tok = pkl.load(x_dev_file)

            print("Loading x_test_process")
            with open(x_test_process, "rb") as x_test_file:
                X_test_tok = pkl.load(x_test_file)

        ## 3. Apply the multilingual model to convert words to their multilingual vectors

        """
        if mode == "multi":
            print ("3. Apply the multilingual model to convert words to multilingual vectors")
            model_path = data_util.emb_model_path + data_util.emb_model_name + "_languages/" + data_util.language + "_vector_model.p"
            if not os.path.isfile(model_path):
                data_util.load_multi_vectors()

            print ("Loading word2vec model for the language")
            with open(data_util.emb_model_path + data_util.emb_model_name + "_languages/" +
                              data_util.language + "_vector_model.p", "rb") as model_file:
                word_vector_dict = pkl.load(model_file)

            ## Apply to train split
            print ("For Train ...")
            x_vec_train = data_util.apply_emb_model(X_train_tok, word_vector_dict)

            ## Apply to dev split
            print ("For Dev ...")
            x_vec_dev = data_util.apply_emb_model(X_dev_tok, word_vector_dict)

            ## Apply to test split
            print ("For Test ...")
            x_vec_test = data_util.apply_emb_model(X_test_tok, word_vector_dict)

        else:
            print("Loading word2vec gensim model")
            if data_util.language == "english" or data_util.language == "german":
                model = KeyedVectors.load_word2vec_format(data_util.emb_model_path + data_util.emb_model_name,
                                                          binary=True)
            else:
                with open(data_util.emb_model_path + data_util.emb_model_name) as vector_file:
                    word_vecs = vector_file.readlines()[1:]

                model = {}
                for word in word_vecs:
                    parts = word.split(" ")
                    model.update({parts[0]: map(float, parts[1:301])})
            ## Apply to train split
            print ("For Train ...")
            x_vec_train = data_util.apply_word2vec_gensim(X_train_tok, model)

            ## Apply to dev split
            print ("For Dev ...")
            x_vec_dev = data_util.apply_word2vec_gensim(X_dev_tok, model)

            ## Apply to test split
            print ("For Test ...")
            x_vec_test = data_util.apply_word2vec_gensim(X_test_tok, model)

        """
        print("Joining ...")
        X_train_tok = [[item for sublist in x_tok for item in sublist] for x_tok in X_train_tok]
        X_dev_tok = [[item for sublist in x_tok for item in sublist] for x_tok in X_dev_tok]
        X_test_tok = [[item for sublist in x_tok for item in sublist] for x_tok in X_test_tok]

        print("len(X_train_tok)=", len(X_train_tok))
        print("len(X_dev_tok)=", len(X_dev_tok))
        print("len(X_test_tok)=", len(X_test_tok))
        ## 4. Create Vocabulary out of the target labels and Look for the significance of the labels #
        # Combining train, dev and test datasets to construct common vocabulary for each language
        print ("4. Creating Vocabulary out of the target labels")
        y = y_train + y_dev + y_test

        label_vocab_path = os.path.join(data_util.data_root, "label_vocab")
        if not os.path.isfile(label_vocab_path):
            label_vocab_list, label_vocab = data_util.create_vocab(label_vocab_path, y)
        else:
            with open(label_vocab_path + ".p", "rb") as file:
                label_vocab = pkl.load(file)

        print ("5. Converting target labels to ids")
        label_ids_train = data_util.label_to_ids_simple(y_train, label_vocab)
        label_ids_dev = data_util.label_to_ids_simple(y_dev, label_vocab)
        label_ids_test = data_util.label_to_ids_simple(y_test, label_vocab)

        print("6 Removing empty and very long documents and their corresponding labels")
        x_vec_train_new = []
        label_ids_train_new = []
        for i in range(0, len(X_train_tok)):
            if 0 < len(X_train_tok[i]) < MAX_SEQUENCES:
                x_vec_train_new.append(X_train_tok[i])
                label_ids_train_new.append(label_ids_train[i])

        x_vec_dev_new = []
        label_ids_dev_new = []
        for i in range(0, len(X_dev_tok)):
            if 0 < len(X_dev_tok[i]) < MAX_SEQUENCES:
                x_vec_dev_new.append(X_dev_tok[i])
                label_ids_dev_new.append(label_ids_dev[i])

        x_vec_test_new = []
        label_ids_test_new = []
        for i in range(0, len(X_test_tok)):
            if 0 < len(X_test_tok[i]) < MAX_SEQUENCES:
                x_vec_test_new.append(X_test_tok[i])
                label_ids_test_new.append(label_ids_test[i])

        self.x_vec_train = x_vec_train_new
        self.label_ids_train = label_ids_train_new
        self.x_vec_dev = x_vec_dev_new
        self.label_ids_dev = label_ids_dev_new
        self.x_vec_test = x_vec_test_new
        self.label_ids_test = label_ids_test_new
