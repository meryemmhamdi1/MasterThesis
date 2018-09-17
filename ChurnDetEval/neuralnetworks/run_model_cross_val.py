import csv
import os

from CrossValDataset import *
from NeuralNets import NeuralNets
from get_args import *
import cPickle as pkl
import numpy as np
from sklearn.model_selection import StratifiedKFold

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["THEANO_FLAGS"] = "/device:GPU:1"

args = get_args()

data_set = args.dataset
max_seq_length = 75 #51 for english only
max_num_words = 1000000 #We don't limit the number of tokens
cross_fold_splits = 10
language = args.language
word_embeddings_path = args.word_embeddings_path
batch_size = args.batch_size
epochs = args.epochs
kernel_sizes = args.kernel_sizes
filters = args.filters
num_of_units = args.num_of_units
dropout = args.dropout
patience = args.patience
network = args.network_type
n_classes = args.n_classes
train_mode = args.train_mode
dataset = args.dataset
embed_name = args.word_embeddings_path.split("/")[-1].split("_")[0]
save_path = "/aimlx/Results/ChurnDet/CGA_CrossVal"+train_mode.upper()+"_EXP_DIC_Keras_Models_"+dataset.upper()

# Path to english and german datasets
data_path = "/aimlx/Datasets/ChurnDet/"

data_en = data_path+'english/x_all.txt'
data_de = data_path + 'german/x_all_text.txt'

data_paths = {"en": data_en, "de": data_de} #{"de": data_de}

data_en_bot = data_path+'english/out_preproc_EN.csv'
data_de_bot = data_path + 'german/out_preproc_DE_new.csv'

data_paths_bot = {"en": data_en_bot, "de": data_de_bot}# {"de": data_de_bot}
directory = "output/"
if not os.path.exists(directory):
    os.makedirs(directory)

# Create the dataset and load the data
dataset = CrossValDataset(data_paths, train_mode, language, word_embeddings_path, max_num_words, max_seq_length)

# Prepare the data for the model
x_dict, labels_dict, y_dict, embedding_matrix, vocab_size, labels_to_ids, vocab = dataset.prepare_data()
inv_vocab = {v: k for k, v in vocab.iteritems()}
#print(x_dict["de"][0])

# Prepare data for bot
dataset_bot = CrossValDataset(data_paths_bot, train_mode, language, word_embeddings_path, max_num_words, max_seq_length)
#x_bot_dict, y_bot_dict, embedding_matrix, vocab, vocab_size = dataset_bot.prepare_data_bot(vocab, labels_to_ids)
x_bot_dict, y_bot_dict = dataset_bot.prepare_data_bot(vocab, labels_to_ids)

#print(x_bot_dict["de"][0:2])
#print(y_bot_dict["de"][0:2])

# The vocab size may be different if the initial max_num_words is greater than the dataset vocabulary
max_num_words = vocab_size

# Initialize the parameters of the network
nn = NeuralNets(filters, kernel_sizes, num_of_units, dropout, patience, embedding_matrix, max_seq_length, max_num_words,
                word_embeddings_path, batch_size, epochs, labels_to_ids)

# Do cross validation here to come up with x_train_dict, y_train_dict and x_test_dict, y_test_dict
print("Preparing Cross Validation Splits ... ")
mean_results = {}
for _ in range(5):
    skf = StratifiedKFold(n_splits=cross_fold_splits, shuffle=True, random_state=0) #cross_fold_splits
    x_train_dict = {}
    y_train_dict = {}
    x_test_dict = {}
    y_test_dict = {}
    for lang in x_dict:
        i = 0
        x_train_fold = {}
        y_train_fold = {}
        x_test_fold = {}
        y_test_fold = {}
        print(y_dict[lang])
        for train_index, test_index in skf.split(x_dict[lang], labels_dict[lang]):
            x_train, x_test = x_dict[lang][train_index], x_dict[lang][test_index]
            y_train, y_test = y_dict[lang][train_index], y_dict[lang][test_index]

            x_train_fold.update({i: x_train})
            y_train_fold.update({i: y_train})
            x_test_fold.update({i: x_test})
            y_test_fold.update({i: y_test})

            """print("Writing the tweets for lang: " + lang + " and fold:" + str(i))
            with open("/aimlx/Results/ChurnDet/x_test_"+lang+"_fold_"+str(i)+".txt", "w") as file:
                for i in range(0, len(x_test)):
                    tweet = ""
                    for id in x_test[i]:
                        if id < len(vocab):
                            tweet += inv_vocab[id].split("_")[1] + " "
                    #print(tweet.encode('utf-8'))
                    if y_test[i][0] == 1:
                        file.write(tweet.encode('utf-8')+"\t 0 \n")
                    else:
                        file.write(tweet.encode('utf-8')+"\t 1 \n")
            """
	    i += 1

            x_train_dict.update({lang: x_train_fold})
            y_train_dict.update({lang: y_train_fold})
            x_test_dict.update({lang: x_test_fold})
            y_test_dict.update({lang: y_test_fold})

    train_metrics_list = []
    test_metrics_list = []
    test_bot_metrics_list = []

    for fold in range(0, cross_fold_splits):
        print("Fold => %i " % fold)

        if "mono" in train_mode:
            train_lang = train_mode.split('_')[0]
            x_train = x_train_dict[train_lang][fold]
            y_train = y_train_dict[train_lang][fold]

            x_test = {train_lang: x_test_dict[train_lang][fold]}
            y_test = {train_lang: y_test_dict[train_lang][fold]}

            x_bot_test = {train_lang: x_bot_dict[lang]}
            y_bot_test = {train_lang: y_bot_dict[lang]}

        else:
            if "," in train_mode:
                train_lang = train_mode.split(',')
            else:
                train_lang = [train_mode]
            x_train = np.concatenate([x_train_dict[train_lang[i]][fold] for i in range(0, len(train_lang))], axis=0)
            y_train = np.concatenate([y_train_dict[train_lang[i]][fold] for i in range(0, len(train_lang))], axis=0)

            x_test = {}
            y_test = {}
            for lang in ["en", "de"]:
                x_test.update({lang: x_test_dict[lang][fold]})
                y_test.update({lang: y_test_dict[lang][fold]})

            x_bot_test = {}
            y_bot_test = {}
            for lang in ["en", "de"]:
                x_bot_test.update({lang: x_bot_dict[lang]})
                y_bot_test.update({lang: y_bot_dict[lang]})

        # Train the model
        acc, results_dict, model = nn.train_model_no_dev(network, x_train, y_train, x_test, y_test, x_bot_test, y_bot_test,
                                                         train_mode, n_classes, train_lang)

        # Save the model per fold
        model.save(save_path + "_fold_" + str(fold) + '.h5')  # creates a HDF5 file 'my_model.h5')

        # Save the results per fold
        with open(save_path + "_fold_" + str(fold) + "_results.p", "wb") as dict_pkl:
            pkl.dump(results_dict, dict_pkl)

        train_metrics = results_dict['train_metrics']
        test_metrics_dict = {}
        for lang in x_test:
            test_metrics_dict.update({lang: results_dict['test_metrics_'+lang]})

        test_bot_metrics_dict = {}
        for lang in x_test:
            test_bot_metrics_dict.update({lang: results_dict['test_bot_metrics_'+lang]})

        train_metrics_list.append(train_metrics)
        test_metrics_list.append(test_metrics_dict)

        test_bot_metrics_list.append(test_bot_metrics_dict)

    # Compute the mean of all test scores
    print("test_metrics_list:", test_metrics_list)
    indices_list = []

    for i in range(0, len(test_metrics_list[-1]["de"])):
        f1_scores = []
        for j in range(0, len(test_metrics_list)):
            f1_scores.append(test_metrics_list[j]["de"][i]["f1_macro"])
        indices_list.append(f1_scores.index(max(f1_scores)))

    print("indices_list =>", indices_list)

    mean_results_dict = {}
    for lang in x_test:
        mean_acc = np.mean([test_metrics_list[indices_list[i]][lang][i]["acc"] for i in range(0, len(test_metrics_list[-1][lang]))])
        mean_f1 = np.mean([test_metrics_list[indices_list[i]][lang][i]["f1_macro"] for i in range(0, len(test_metrics_list[-1][lang]))])
	max_f1 = np.max([test_metrics_list[indices_list[i]][lang][i]["f1_macro"] for i in range(0, len(test_metrics_list[-1][lang]))])
        mean_pre = np.mean([test_metrics_list[indices_list[i]][lang][i]["precision_macro"] for i in range(0, len(test_metrics_list[-1][lang]))])
        mean_rec = np.mean([test_metrics_list[indices_list[i]][lang][i]["recall_macro"] for i in range(0, len(test_metrics_list[-1][lang]))])
        print("Mean test accuracy for lang %s => %f " % (lang, mean_acc))
        print("Mean test f1 score for lang %s => %f " % (lang, mean_f1))
        print("Mean test precision for lang %s => %f " % (lang, mean_pre))
        print("Mean test recall for lang %s => %f " % (lang, mean_rec))
        if lang not in mean_results:
            mean_results.update({lang:[]})

        list_results = mean_results[lang]
        list_results.append(max_f1)
        mean_results.update({lang: list_results})
	
	with open("/aimlx/mean_results.txt", "a") as temp_file:
		temp_file.write(lang+ " => " + str(max_f1))
		
        mean_test_results = {"acc": mean_acc, "f1_macro": mean_f1, "precision_macro": mean_pre, "recall_macro": mean_rec}
        mean_results_dict['test_metrics_'+lang] = mean_test_results

    # Compute the mean of all train scores
    mean_acc = np.mean([train_metrics_list[i][indices_list[i]]["acc"] for i in range(0, len(train_metrics_list))])
    mean_f1 = np.mean([train_metrics_list[i][indices_list[i]]["f1_macro"] for i in range(0, len(train_metrics_list))])
    mean_pre = np.mean([train_metrics_list[i][indices_list[i]]["precision_macro"] for i in range(0, len(train_metrics_list))])
    mean_rec = np.mean([train_metrics_list[i][indices_list[i]]["recall_macro"] for i in range(0, len(train_metrics_list))])
    print("Mean train accuracy => ", mean_acc)
    print("Mean train f1 score => ", mean_f1)
    print("Mean train precision => ", mean_pre)
    print("Mean train recall => ", mean_rec)

    mean_train_results = {"acc": mean_acc, "f1_macro": mean_f1, "precision_macro": mean_pre, "recall_macro": mean_rec}
    mean_results_dict["train_metrics"] = mean_train_results

    # Save the results per fold
    #with open(save_path + "_mean_results.p", "wb") as dict_pkl:
        #pkl.dump(mean_results_dict, dict_pkl)

with open(save_path + "_2_RUNS_results.p", "wb") as dict_pkl:
	pkl.dump(mean_results, dict_pkl)

