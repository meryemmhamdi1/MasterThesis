import csv
import os

from Dataset import *
from NeuralNets import NeuralNets
from get_args import *
import cPickle as pkl
import numpy as np

from sklearn.model_selection import StratifiedKFold
StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

args = get_args()

data_set = args.dataset
max_seq_length = 75 # 51 for english only
max_num_words = 1000000  # We don't limit the number of tokens
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
save_path = "/aimlx/Results/ChurnDet/GA_"+train_mode.upper()+"_BILINGUAL_Keras_Models_"+dataset.upper()


# Path to train and test set
data_path = "/aimlx/Datasets/ChurnDet/"#"/aimlx/RCV1_RCV2/ECGA/"
train_en = data_path+'english/train.txt'
dev_en = data_path + 'english/dev.txt'
test_en = data_path + 'english/test.txt'

train_de = data_path + 'german/train.txt'
dev_de = data_path + 'german/dev.txt'
test_de = data_path + 'german/test.txt'

"""
train_fr = data_path + 'french/train.txt'
dev_fr = data_path + 'french/dev.txt'
test_fr = data_path + 'french/test.txt'

train_it = data_path + 'italian/train.txt'
dev_it = data_path + 'italian/dev.txt'
test_it = data_path + 'italian/test.txt'

"""
train = {"en":train_en, "de": train_de} #, "fr": train_fr, "it": train_it}
dev = {"en": dev_en, "de": dev_de} #, "fr": dev_fr, "it": dev_it} ,
test = {"en": test_en, "de": test_de}#, }# , "fr": test_fr, "it": test_it}

if "mono" in train_mode:
    train_lang = train_mode.split('_')[0]
    train = dict((key, value) for key, value in train.iteritems() if key == train_lang)
    dev = dict((key, value) for key, value in dev.iteritems() if key == train_lang)
    test = dict((key, value) for key, value in test.iteritems() if key == train_lang)

directory = "output/"
if not os.path.exists(directory):
    os.makedirs(directory)

# Create the dataset and load the data
dataset = Dataset(train, dev, test, train_mode, language, word_embeddings_path, max_num_words, max_seq_length)

# Prepare the data for the model
x_train_dict, y_train_dict, x_dev_dict, y_dev_dict, x_test_dict, y_test_dict, \
embedding_matrix, vocab_size, labels_to_ids = dataset.prepare_data()

# The vocab size may be different if the initial max_num_words is greater than the dataset vocabulary
max_num_words = vocab_size

# Initialize the parameters of the network
nn = NeuralNets(filters, kernel_sizes, num_of_units, dropout, patience, embedding_matrix, max_seq_length, max_num_words,
                word_embeddings_path, batch_size, epochs, labels_to_ids)

# Train the network and return scores (F-score or accuracy)
if "mono" in train_mode:
    train_lang = train_mode.split('_')[0]
    x_train = x_train_dict[train_lang]
    y_train = y_train_dict[train_lang]

    x_dev = x_dev_dict[train_lang]
    y_dev = y_dev_dict[train_lang]

    x_test = {train_lang:x_test_dict[train_lang]}
    y_test = {train_lang:y_test_dict[train_lang]}

else:
    if "," in train_mode:
        train_lang = train_mode.split(',')
    else:
        train_lang = [train_mode]
    x_train = np.concatenate([x_train_dict[train_lang[i]] for i in range(0, len(train_lang))], axis=0)
    y_train = np.concatenate([y_train_dict[train_lang[i]] for i in range(0, len(train_lang))], axis=0)
    x_dev = np.concatenate([x_dev_dict[train_lang[i]] for i in range(0, len(train_lang))], axis=0)
    y_dev = np.concatenate([y_dev_dict[train_lang[i]] for i in range(0, len(train_lang))], axis=0)

    x_test = x_test_dict
    y_test = y_test_dict

acc, results_dict, model = nn.train_model(network, x_train, y_train, x_dev, y_dev, x_test, y_test, train_mode, n_classes, train_lang)

model.save(save_path + '.h5')  # creates a HDF5 file 'my_model.h5')

with open(save_path + "_results.p", "wb") as dict_pkl:
    pkl.dump(results_dict, dict_pkl)

# Write results to file
with open(directory + 'table_' + network + "_" + data_set + '.csv', 'a') as csv_file:
    writer = csv.writer(csv_file, delimiter=",", quoting=csv.QUOTE_MINIMAL)
    writer.writerow(
        ['network_type', 'max_seq_length', 'max_num_words', 'word_embeddings_path', 'batch_size',
         'max_epochs', 'kernel_sizes', 'filters', 'num_of_units', 'dropout', 'data_set', 'patience', 'max_acc'])
    writer.writerow(
        [network, max_seq_length, max_num_words, args.word_embeddings_path, args.batch_size, args.epochs,
         args.kernel_sizes, args.filters, args.num_of_units, args.dropout, dataset, args.patience, acc])
