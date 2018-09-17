import cPickle as pkl
import os
import sys

sys.path.insert(0, '/aimlx/MultiEmb_EventDet_Thesis/dataModule/')
# sys.path.insert(0, '/Users/MeryemMhamdi/EPFL/Spring2018/Thesis/3 Algorithms Implementation/MultiEmb_EventDet_Thesis/dataModule/')
import argparse
from xml.dom import minidom
import random
import numpy as np
from sklearn.manifold import TSNE
import json

SHUFFLE_SEED = 100099540
TRAIN_PERC = 0.6
dev_PERC = 0.2
TEST_PERC = 0.2
NB_SAMPLES_CLASS = 100


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


class TestRCVProcessor(object):
    def __init__(self, data_util):
        x_test_p = data_util.pre_dir + "test/X_test_" + data_util.language + ".p"
        x_test_process = data_util.pre_dir + "test/X_test_processed_" + data_util.language + ".p"
        y_test_p = data_util.pre_dir + "test/y_test_" + data_util.language + ".p"

        print("Reading from x_test_p: %s ..." % x_test_p)
        with open(x_test_process, "rb") as x_test_file:
            X_test = pkl.load(x_test_file)

        print("Reading from x_test_process: %s ..." % x_test_process)
        with open(x_test_process, "rb") as x_test_file:
            X_test_tok = pkl.load(x_test_file)

        print("Reading from y_test_p: %s ..." % y_test_p)
        with open(y_test_p, "rb") as y_test_file:
            y_test = pkl.load(y_test_file)

        # 3. Apply the multilingual model to convert words to their multilingual vectors
        print ("3. Apply the multilingual model to convert words to multilingual vectors")
        model_path = data_util.emb_model_path + data_util.emb_model_name + "_languages/" + data_util.language + "_vector_model.p"
        if not os.path.isfile(model_path):
            data_util.load_multi_vectors()

        print ("Loading word2vec model for the language")
        with open(data_util.emb_model_path + data_util.emb_model_name + "_languages/" +
                          data_util.language + "_vector_model.p", "rb") \
                as model_file:
            word_vector_dict = pkl.load(model_file)

        x_vec_test = data_util.apply_emb_model(X_test_tok, word_vector_dict)

        #print("X_test_tok[0]", X_test_tok[0])
        #print("x_vec_test[0] ", x_vec_test[0])

        x_vec_test_new = []
        X_test_new = []
        y_test_new = []
        for i in range(0, len(x_vec_test)):
            if len(x_vec_test[i]) > 0:
                x_vec_test_new.append(x_vec_test[i])
                X_test_new.append(X_test[i])
                y_test_new.append(y_test[i])

        x_test_avg = data_util.sent_avg(x_vec_test_new)

        #print(" x_test_avg[0] ", x_test_avg[0])

        test_dict_list = []
        for i in range(0, len(x_test_avg)):
            test_dict_list.append({'text': X_test_new[i], 'vec': x_test_avg[i], 'true_label': y_test_new[i]})

        self.x_vec_test = test_dict_list
        self.label_ids_test = y_test_new


def get_args_gpu():
    """
    Get command line arguments.

    Arguments set the default values of command line arguments.
    """
    parser = argparse.ArgumentParser()

    """Dataset Path Parameters"""
    parser.add_argument("--data-choice", "-dc", type=str, default="rcv",
                        help='Choice of the dataset to be used for Crosslingual Document Classification: '
                             'dw for DeutscheWelle'
                             'rcv for Reuters Dataset'
                             'ted for TED Corpus')

    parser.add_argument("--data-dw", "-ddw", type=str,
                        default="/aimlx/mhan/common_data/dw_general/",
                        help='The higher level directory path of the DeutscheWelle train/dev/test dataset')

    parser.add_argument("--data-ted", "-dted", type=str,
                        default="/aimlx/TED/",
                        help='The higher level directory path of TED Corpus')

    parser.add_argument("--data-rcv1", "-drcv1", type=str,
                        default="/aimlx/rcv1/",
                        help='The higher level directory path of the RCV1')

    parser.add_argument("--data-rcv2", "-drcv2", type=str,
                        default="/aimlx/RCV2_Multilingual_Corpus/",
                        help='The higher level directory path of the RCV2')

    parser.add_argument("--pre-dir", "-fd", type=str,
                        default="processed/",
                        help='Directory of Cross Fold Validation Dataset in the form of csv files (one for each language)')

    """Parameters and Files related to Embedding models and pickles files"""
    ## Monolingual Models
    parser.add_argument("--w2v-dir", "-w2dir", type=str,
                        default="/aimlx/MonolingualEmbeddings/",
                        help='Path of monolingual word vector models')

    parser.add_argument("--w2v-en", "-w2en", type=str,
                        default="english.pkl",
                        help='Path to the pickle file with the list of vectors for words in the input vocabulary in English')

    parser.add_argument("--w2v-de", "-w2de", type=str,
                        default="german.pkl",
                        help='Path to the pickle file with the list of vectors for words in the input vocabulary in Deutsche')

    ## Multilingual Models
    parser.add_argument("--w2v-multi-choice", "-w2v-choice", type=str,
                        default="",
                        help='Choice of the multilingual embedding model')

    parser.add_argument("--model_dir", "-md", type=str,
                        default="/aimlx/MultilingualEmbeddings/",
                        help='Path to multilingual word vector models')

    parser.add_argument("--multi-skipgram", "-msg", type=str, default="multiSkip_40_normalized",
                        help='Multi-skip gram model')

    parser.add_argument("--multi-cluster", "-mc", type=str, default="",
                        help='Path to the multilingual word embeddings trained using joint model BIBLOWA')

    parser.add_argument("--embed-dim", "-ed", type=int, default=40, help="Embedding dimension")

    """Parameters related to Pre-processing"""
    parser.add_argument("--stop-pos-path", "-spp", type=str, default="",
                        help='Path of the file that lists the pos tags of the words that need to be removed')

    parser.add_argument("--lemma-use", "-lu", type=bool, default=False,
                        help='Whether to use lemmatization or not')

    parser.add_argument("--stop-use", "-su", type=bool, default=False,
                        help='Whether to use stopword removal or not')

    """Parameters related to training of the model"""
    parser.add_argument("--model-choice", "-mdc", type=str, default="cnn", help='the choice of model')

    parser.add_argument("--languages", "-langs", type=str, default="english,german,french,italian",
                        help='the list of languages separated by comma')

    parser.add_argument("--multi-train", "-mt", type=str, default="en",
                        help="Choice of the multilingual mode of training the model:"
                             "en: for English only"
                             "de: for German only"
                             "fr: for French only"
                             "it: for Italian only"
                             "en,de: for English-Deutsch"
                             "en,fr: for English-French"
                             "en,it: for English-Italian"
                             "fr,de: for French-Deutsch"
                             "fr,it: for French-Italian"
                             "de,it: for Deutsch-Italian"
                             "en,fr,de: for English-French-Deutsch"
                             "en,fr,it: for English-French-Italian"
                             "fr,it,de: for French-Italian-Deutsch"
                             "en,de,fr,it: all languages")

    parser.add_argument("--batch_size", "-bs", type=int, default=32, help='the size of minibatch')

    parser.add_argument("--epochs", "-ep", type=int, default=100, help='The number of epochs used to train the model')

    parser.add_argument("--filter-sizes", "-fs", type=str, default="3,4,5",
                        help="The size of each filter in the multi-filter CNN")
    parser.add_argument("--num-filters", "-nf", type=int, default=300, help="Number of feature maps per filter type")
    parser.add_argument("--dropout-per", "-dp", type=float, default=0.5, help="Percentage of dropout")

    parser.add_argument("--model-save-path", "-rmp", type=str, default="/aimlx/MLP_Keras_Models_RCV/",
                        help='The root path where the model should be saved')

    parser.add_argument("--model_file", "-mf", type=str, default="model_no_val.yaml",
                        help='The path where the model should be saved')

    parser.add_argument("--model-weights-path", "-mwp", type=str, default="model_no_val.h5",
                        help='The path where the model should be saved with its weights')

    return parser.parse_args()



if __name__ == '__main__':
    global args, lang_list, lang_dict

    args = get_args_gpu()

    lang_list = args.languages.split(',')

    lang_dict = {}
    with open("../iso_lang_abbr.txt") as iso_lang:
        for line in iso_lang:
            lang_dict.update({line.split(":")[1][:-1]: line.split(":")[0]})

    print(lang_dict)

    class_symbols = {'GCAT': 'circle', 'MCAT': 'square', 'CCAT': 'diamond', 'ECAT': 'cross'}

    json_dict = {}
    tsne = TSNE(n_components=2, random_state=0)
    for language in lang_list:
        # print("Processing ",language)
        # if language == "english":
        #     data_util = data_utils.DataUtils(args.data_rcv1, args.data_rcv1 + args.pre_dir, args.stop_pos_path,
        #                                      args.lemma_use,
        #                                      args.stop_use, language, args.model_dir, args.multi_skipgram,
        #                                      args.embed_dim)
        #
        # else:
        #     data_util = data_utils.DataUtils(args.data_rcv2 + language + "/",
        #                                      args.data_rcv2 + language + "/" + args.pre_dir,
        #                                      args.stop_pos_path, args.lemma_use, args.stop_use, language,
        #                                      args.model_dir,
        #                                      args.multi_skipgram, args.embed_dim)
        #
        # dp = TestRCVProcessor(data_util)
        # x_test = dp.x_vec_test
        #
        # with open(args.data_rcv1 + args.pre_dir + 'test_dict_' + language + '.p', "wb") as pkl_file:
        #     pkl.dump(x_test, pkl_file)

        print("Processing => ", language)
        ## 1. Reading the pickle file for the language which contains a list of dictionaries with text, vec, true_label
        print(" 1. Reading the pickle file for the language")
        with open(args.data_rcv1 + args.pre_dir + 'test_dict_' + language + '.p') as pkl_file:
            test_dict_list = pkl.load(pkl_file)

        ## 2. Read the predicted labels and true labels using MLP trained on all languages
        print(" 2. Read the predicted labels and true labels using MLP trained on all languages")
        with open(args.model_save_path + "en,de,fr,it_multiSkip_40_normalized_results.p") as file:
            results = pkl.load(file)

        ## 3. Map label ids to labels texts
        print(" 3. Map label ids to labels texts")
        with open(args.data_rcv1 + args.pre_dir+"label_vocab.p") as labels_file:
            labels_dict = pkl.load(labels_file)
        labels_inv = {v: k for k, v in labels_dict.iteritems()}

        pred_text_list = []
        pred_results = results['y_test_pred_'+lang_dict[language]]
        for pred in pred_results:
            pred_text_list.append(labels_inv[pred])

        ## 4. Iterate through the list to get the list per class
        print(" 4. Iterate through the list to get the list per class")
        test_dict_list_class = {}
        for i in range(0, len(test_dict_list)):
            if pred_text_list[i] in test_dict_list_class:
                values = test_dict_list_class[pred_text_list[i]]
                values.append({'text': test_dict_list[i]['text'], 'vec': test_dict_list[i]['vec']})
                test_dict_list_class.update({pred_text_list[i]: values})
            else:
                values = [{'text': test_dict_list[i]['text'], 'vec':test_dict_list[i]['vec']}]
                test_dict_list_class.update({pred_text_list[i]: values})

        ## 5. Select 25 instances randomly from each class then merge
        print(" 5. Select a sample of 25 instances randomly")
        random.seed(SHUFFLE_SEED)
        test_dict_list_class_new = {}
        for class_ in test_dict_list_class:
            print("class_=", class_)
            myList = test_dict_list_class[class_]
            random.shuffle(myList)
            print("len(myList)=", len(myList))
            if len(myList) > NB_SAMPLES_CLASS:
                test_dict_list_class_new.update({class_: random.sample(myList, NB_SAMPLES_CLASS)})
            else:
                test_dict_list_class_new.update({class_: myList})


        text_list = []
        vec_40_list = []
        pred_list = []
        symbol_list = []
        for class_ in test_dict_list_class_new:
            for item_ in test_dict_list_class_new[class_]:
                text_list.append(" ".join([item for sublist in item_['text'] for item in sublist]))
                vec_40_list.append(item_['vec'])
                pred_list.append(class_)
                symbol_list.append(class_symbols[class_])

        #print("vec_40_list= ",vec_40_list)
        arr = np.array(vec_40_list)
        Y = tsne.fit_transform(arr)
        x = list(Y[:, 0].astype(np.float))
        y = list(Y[:, 1].astype(np.float))

        ## 6. Save the list of elements

        print(" 6. Dumping for language")
        # json_dict.update({lang_dict[language]: {'avg': vec_40_list, 'mode': 'markers', 'type': 'scatter', 'text': text_list,
        #                                         'marker': {'size': 20, 'symbol': symbol_list}}})
        json_dict.update({lang_dict[language]: {'x': x, 'y': y, 'mode': 'markers', 'type': 'scatter', 'text': text_list,
                                                'marker': {'size': 20, 'symbol': symbol_list}}})

    # ## 7. Apply TSNE
    # print(" 7. Applying TSNE")
    # vectors_all_lang = []
    # for lang in json_dict:
    #     vectors_all_lang.append(json_dict[lang]['avg'])
    #
    # vectors_all = [item for sublist in vectors_all_lang for item in sublist]
    # arr = np.array(vectors_all)
    # tsne = TSNE(n_components=2, random_state=0)
    # np.set_printoptions(suppress=True)
    # Y = tsne.fit_transform(arr)
    # x = list(Y[:, 0].astype(np.float))
    # y = list(Y[:, 1].astype(np.float))
    #
    # begin = 0
    # for lang in json_dict:
    #     x_lang = x[begin:begin+len(json_dict[lang]['text'])]
    #     y_lang = y[begin:begin+len(json_dict[lang]['text'])]
    #     json_dict.update({lang: {'x': x_lang, 'y': y_lang, 'mode': 'markers', 'type': 'scatter', 'text': json_dict[lang]['text'],
    #                              'marker': {'size': 20, 'symbol': json_dict[lang]['marker']['symbol']}}})
    #     begin = begin + len(json_dict[lang]['text'])
    ## 8. Save the pickle file:
    print(" 8. Saving pickle file")
    with open(args.data_rcv1 + args.pre_dir+"all_languages_mlp_demo.json", "w") as file:
        json.dump(json_dict, file)
