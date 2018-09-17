import os
import cPickle as pkl
import glob
from RCV_doc import *
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from xml.dom import minidom
from tqdm import tqdm


SHUFFLE_SEED = 100099540
TRAIN_PERC = 0.6
DEV_PERC = 0.2
TEST_PERC = 0.2
MAX_SEQUENCES = 622


def process_xml_rcv(file_):

    xmldoc = minidom.parse(file_)
    itemid = xmldoc.getElementsByTagName('newsitem')[0].attributes['itemid'].firstChild.data
    date = xmldoc.getElementsByTagName('newsitem')[0].attributes['date'].firstChild.data
    lang = xmldoc.getElementsByTagName('newsitem')[0].attributes['xml:lang'].firstChild.data
    # title = xmldoc.getElementsByTagName('title')[0].firstChild.data
    text = xmldoc.getElementsByTagName('text')[0].getElementsByTagName('p')

    #texts = [title]
    texts = []
    for i in range(0,len(text)):
        texts.append(text[i].firstChild.data)

    classes = ['bip:countries:1.0', 'bip:industries:1.0', 'bip:topics:1.0']

    codes = xmldoc.getElementsByTagName('metadata')[0].getElementsByTagName('codes')
    countries = []
    topics = []
    industries = []
    for i in range(0,len(codes)):
        class_ = codes[i].attributes['class'].value
        if class_ == classes[0]: # 'bip:countries:1.0'
            codes_class = codes[i].getElementsByTagName('code')
            for j in range(0,len(codes_class)):
                countries.append(codes_class[j].attributes['code'].firstChild.data)
        elif class_ == classes[1]: # 'bip:industries:1.0'
            codes_class = codes[i].getElementsByTagName('code')
            for j in range(0,len(codes_class)):
                industries.append(codes_class[j].attributes['code'].firstChild.data)
        else:
            codes_class = codes[i].getElementsByTagName('code')
            for j in range(0,len(codes_class)):
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


class KerasProcessor(object):

    def __init__(self, data_util, lang_dict):

        x_train_new_p = data_util.pre_dir + "train/X_train_new_" + data_util.language + ".p"
        x_dev_new_p = data_util.pre_dir + "dev/X_dev_new_" + data_util.language + ".p"
        x_test_new_p = data_util.pre_dir + "test/X_test_new_" + data_util.language + ".p"

        label_ids_train_new_p = data_util.pre_dir + "train/label_ids_train_new_" + data_util.language + ".p"
        label_ids_dev_new_p = data_util.pre_dir + "dev/label_ids_dev_new_" + data_util.language + ".p"
        label_ids_test_new_p = data_util.pre_dir + "test/label_ids_test_new_" + data_util.language + ".p"

        if not os.path.isfile(x_train_new_p) or not os.path.isfile(x_dev_new_p) or not os.path.isfile(x_test_new_p) or \
            not os.path.isfile(label_ids_train_new_p) or not os.path.isfile(label_ids_dev_new_p) or not os.path.isfile(label_ids_test_new_p):

            ## Apply Preprocessing: tokenization, lemmatization and stop words removal
            print("Load or Apply Preprocessing: tokenization, lemmatization and stop words removal")
            # x_train_process = data_util.pre_dir + "train/X_train_processed_" + data_util.language + ".p"
            # x_dev_process = data_util.pre_dir + "dev/X_dev_processed_" + data_util.language + ".p"
            # x_test_process = data_util.pre_dir + "test/X_test_processed_" + data_util.language + ".p"

            x_train_process = data_util.pre_dir + "train/X_train_processed_joined_" + data_util.language + ".p"
            x_dev_process = data_util.pre_dir + "dev/X_dev_processed_joined_" + data_util.language + ".p"
            x_test_process = data_util.pre_dir + "test/X_test_processed_joined_" + data_util.language + ".p"

            label_ids_train_p = data_util.pre_dir + "train/label_ids_train_" + data_util.language + ".p"
            label_ids_dev_p = data_util.pre_dir + "dev/label_ids_dev_" + data_util.language + ".p"
            label_ids_test_p = data_util.pre_dir + "test/label_ids_test_" + data_util.language + ".p"
    
            if not os.path.isfile(x_train_process) or not os.path.isfile(x_dev_process) or not os.path.isfile(x_test_process):
                print(" 1. Loading x_train, y_train, x_dev, y_dev, x_test, y_test")
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
                    print("1. Reading xml files and gathering them into one big list of x features and their corresponding labels")
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
    
                    x_train, x_test, y_train, y_test = \
                        train_test_split(texts, lead_topics, test_size=TEST_PERC, random_state=SHUFFLE_SEED)
    
                    x_train, x_dev, y_train, y_dev = \
                        train_test_split(x_train, y_train, test_size=DEV_PERC, random_state=SHUFFLE_SEED)
    
                    if not os.path.isdir(data_util.pre_dir + "train/"):
                        os.makedirs(data_util.pre_dir + "train/")
    
                    if not os.path.isdir(data_util.pre_dir + "dev/"):
                        os.makedirs(data_util.pre_dir + "dev/")
    
                    if not os.path.isdir(data_util.pre_dir + "test/"):
                        os.makedirs(data_util.pre_dir + "test/")
    
                    with open(x_train_p, "wb") as x_train_file:
                        pkl.dump(x_train, x_train_file)
    
                    with open(y_train_p, "wb") as y_train_file:
                        pkl.dump(y_train, y_train_file)
    
                    with open(x_dev_p, "wb") as x_dev_file:
                        pkl.dump(x_dev, x_dev_file)
    
                    with open(y_dev_p, "wb") as y_dev_file:
                        pkl.dump(y_dev, y_dev_file)
    
                    with open(x_test_p, "wb") as x_test_file:
                        pkl.dump(x_test, x_test_file)
    
                    with open(y_test_p, "wb") as y_test_file:
                        pkl.dump(y_test, y_test_file)
    
                else:
                    # # Reading pickle files
                    # with open(x_train_p, "rb") as x_train_file:
                    #     x_train = pkl.load(x_train_file)

                    print("Loading y_train")
                    with open(y_train_p, "rb") as y_train_file:
                        y_train = pkl.load(y_train_file)
    
                    # with open(x_dev_p, "rb") as x_dev_file:
                    #     x_dev = pkl.load(x_dev_file)

                    print("Loading y_dev")
                    with open(y_dev_p, "rb") as y_dev_file:
                        y_dev = pkl.load(y_dev_file)
    
                    # with open(x_test_p, "rb") as x_test_file:
                    #     x_test = pkl.load(x_test_file)

                    print("Loading y_test")
                    with open(y_test_p, "rb") as y_test_file:
                        y_test = pkl.load(y_test_file)
                #####
                # x_train_tok = data_util.nltk_tokenizer(x_train, "train")
                #
                # x_dev_tok = data_util.nltk_tokenizer(x_dev, "dev")
                #
                # x_test_tok = data_util.nltk_tokenizer(x_test, "test")

                print("Loading x_train_process")
                with open(x_train_process, "rb") as x_train_file:
                    x_train_tok = pkl.load(x_train_file)

                print("Loading x_dev_process")
                with open(x_dev_process, "rb") as x_dev_file:
                    x_dev_tok = pkl.load(x_dev_file)

                print("Loading x_test_process")
                with open(x_test_process, "rb") as x_test_file:
                    x_test_tok = pkl.load(x_test_file)
    
                print("3. Joining the documents Train")
                x_train_tok_join = [[item for sublist in x_train_tok[i] for item in sublist]
                                    for i in tqdm(range(0, len(x_train_tok)))]
                print("Joining documents in Dev")
                x_dev_tok_join = [[item for sublist in x_dev_tok[i] for item in sublist]
                                  for i in tqdm(range(0, len(x_dev_tok)))]

                print("Joining documents in Test")
                x_test_tok_join = [[item for sublist in x_test_tok[i] for item in sublist]
                                   for i in tqdm(range(0, len(x_test_tok)))]

                print("4. Removing Stop words for Train")
                x_train_tok_joined = []
                for ind in tqdm(range(0, len(x_train_tok_join))):
                    """x_train_tok_joined.append([lang_dict[data_util.language]+"_"+word for word in x_train_tok_join[ind] if
                                               word not in stopwords.words(data_util.language)]) """
                    x_train_tok_joined.append([word for word in x_train_tok_join[ind] if
                                               word not in stopwords.words(data_util.language)])

                print("4. Removing Stop words for Dev")
                x_dev_tok_joined = []
                for ind in tqdm(range(0, len(x_dev_tok_join))):
                    """x_dev_tok_joined.append([lang_dict[data_util.language]+"_"+word for word in x_dev_tok_join[ind] if
                                             word not in stopwords.words(data_util.language)]) """
                    x_dev_tok_joined.append([word for word in x_dev_tok_join[ind] if
                                             word not in stopwords.words(data_util.language)])

                print("4. Removing Stop words for Test")
                x_test_tok_joined = []
                for ind in tqdm(range(0, len(x_test_tok_join))):
                    """ x_test_tok_joined.append([lang_dict[data_util.language]+"_"+word for word in x_test_tok_join[ind] if
                                              word not in stopwords.words(data_util.language)])"""
                    x_test_tok_joined.append([word for word in x_test_tok_join[ind] if
                                              word not in stopwords.words(data_util.language)])

                print("Saving New Version of x_train after stop word removal")
                with open(x_train_process, "wb") as x_train_file:
                    pkl.dump(x_train_tok_joined, x_train_file)

                print("Saving New Version of x_dev after stop word removal")
                with open(x_dev_process, "wb") as x_dev_file:
                    pkl.dump(x_dev_tok_joined, x_dev_file)

                print("Saving New Version of x_test after stop word removal")
                with open(x_test_process, "wb") as x_test_file:
                    pkl.dump(x_test_tok_joined, x_test_file)
    
                # 4. Create Vocabulary out of the target labels and Look for the significance of the labels #
                # Combining train, dev and test datasets to construct common vocabulary for each language
                print ("4. Creating Vocabulary out of the target labels")
                y = y_train + y_dev + y_test
    
                label_vocab_path = os.path.join(data_util.data_root, "label_vocab")
                if not os.path.isfile(label_vocab_path):
                    label_vocab_list, label_vocab = data_util.create_vocab(label_vocab_path, y)
                else:
                    with open(label_vocab_path + ".p", "rb") as file_vocab:
                        label_vocab = pkl.load(file_vocab)
    
                print ("5. Converting target labels to ids")
                label_ids_train = data_util.label_to_ids_simple(y_train, label_vocab)
                label_ids_dev = data_util.label_to_ids_simple(y_dev, label_vocab)
                label_ids_test = data_util.label_to_ids_simple(y_test, label_vocab)
    
                print("Saving label_ids_train")
                with open(label_ids_train_p, "wb") as label_ids_train_file:
                    pkl.dump(label_ids_train, label_ids_train_file)
    
                print("Saving label_ids_dev")
                with open(label_ids_dev_p, "wb") as label_ids_dev_file:
                    pkl.dump(label_ids_dev, label_ids_dev_file)
    
                print("Saving label_ids_test")
                with open(label_ids_test_p, "wb") as label_ids_test_file:
                    pkl.dump(label_ids_test, label_ids_test_file)

            else:
                print("Loading x_train_process")
                with open(x_train_process, "rb") as x_train_file:
                    x_train_tok_joined = pkl.load(x_train_file)
    
                print("Loading x_dev_process")
                with open(x_dev_process, "rb") as x_dev_file:
                    x_dev_tok_joined = pkl.load(x_dev_file)
    
                print("Loading x_test_process")
                with open(x_test_process, "rb") as x_test_file:
                    x_test_tok_joined = pkl.load(x_test_file)
    
                print("Loading label_ids_train")
                with open(label_ids_train_p, "rb") as label_ids_train_file:
                    label_ids_train = pkl.load(label_ids_train_file)
    
                print("Saving label_ids_dev")
                with open(label_ids_dev_p, "rb") as label_ids_dev_file:
                    label_ids_dev = pkl.load(label_ids_dev_file)
    
                print("Saving label_ids_test")
                with open(label_ids_test_p, "rb") as label_ids_test_file:
                    label_ids_test = pkl.load(label_ids_test_file)
    
            print("6 Removing empty and very long documents and their corresponding labels")
            x_train_new = []
            label_ids_train_new = []
            for i in range(0, len(x_train_tok_joined)):
                if 0 < len(x_train_tok_joined[i]) < MAX_SEQUENCES:
                    x_train_new.append(x_train_tok_joined[i])
                    label_ids_train_new.append(label_ids_train[i])
    
            x_dev_new = []
            label_ids_dev_new = []
            for i in range(0, len(x_dev_tok_joined)):
                if 0 < len(x_dev_tok_joined[i]) < MAX_SEQUENCES:
                    x_dev_new.append(x_dev_tok_joined[i])
                    label_ids_dev_new.append(label_ids_dev[i])
    
            x_test_new = []
            label_ids_test_new = []
            for i in range(0, len(x_test_tok_joined)):
                if 0 < len(x_test_tok_joined[i]) < MAX_SEQUENCES:
                    x_test_new.append(x_test_tok_joined[i])
                    label_ids_test_new.append(label_ids_test[i])

            print("Saving x_train_new_p")
            with open(x_train_new_p, "wb") as x_train_file:
                pkl.dump(x_train_new, x_train_file)

            print("Saving label_ids_train_new_p")
            with open(label_ids_train_new_p, "wb") as y_train_file:
                pkl.dump(label_ids_train_new, y_train_file)

            print("Saving x_dev_new_p")
            with open(x_dev_new_p, "wb") as x_dev_file:
                pkl.dump(x_dev_new, x_dev_file)

            print("Saving label_ids_dev_new_p")
            with open(label_ids_dev_new_p, "wb") as y_dev_file:
                pkl.dump(label_ids_dev_new, y_dev_file)

            print("Saving x_test_new_p")
            with open(x_test_new_p, "wb") as x_test_file:
                pkl.dump(x_test_new, x_test_file)

            print("Saving label_ids_test_new_p")
            with open(label_ids_test_new_p, "wb") as y_test_file:
                pkl.dump(label_ids_test_new, y_test_file)
                    
        else:
            print("Loading x_train_new_p")
            with open(x_train_new_p, "rb") as x_train_file:
                x_train_new = pkl.load(x_train_file)

            print("Loading label_ids_train_new_p")
            with open(label_ids_train_new_p, "rb") as y_train_file:
                label_ids_train_new = pkl.load(y_train_file)

            print("Loading x_dev_new_p")
            with open(x_dev_new_p, "rb") as x_dev_file:
                x_dev_new = pkl.load(x_dev_file)

            print("Loading label_ids_dev_new_p")
            with open(label_ids_dev_new_p, "rb") as y_dev_file:
                label_ids_dev_new = pkl.load(y_dev_file)

            print("Loading x_test_new_p")
            with open(x_test_new_p, "rb") as x_test_file:
                x_test_new = pkl.load(x_test_file)

            print("Loading label_ids_test_new_p")
            with open(label_ids_test_new_p, "rb") as y_test_file:
                label_ids_test_new = pkl.load(y_test_file)

        self.x_train_doc = x_train_new
        self.x_dev_doc = x_dev_new
        self.x_test_doc = x_test_new
        self.label_ids_train = label_ids_train_new
        self.label_ids_dev = label_ids_dev_new
        self.label_ids_test = label_ids_test_new



