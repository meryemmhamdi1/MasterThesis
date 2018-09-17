model_dir = "/aimlx/Results/ChurnDet/CGA_CrossValEN,DE_SEM_SPEC_Keras_Models_TUNED_CHURN_fold_5.h5"

from keras.models import load_model
import csv
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from gensim.models import KeyedVectors
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from string import punctuation
import cPickle as pkl
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import csv
import os

from CrossValDataset import *
from NeuralNets import NeuralNets
from get_args import *
import cPickle as pkl
import numpy as np
from sklearn.model_selection import StratifiedKFold

print("Load model")
model = load_model(model_dir)

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
save_path = "/aimlx/Results/ChurnDet/CGA_CrossVal"+train_mode.upper()+"_SEM_SPEC_Keras_Models_TUNED_"+dataset.upper()

# Path to english and german datasets
data_path = "/aimlx/Datasets/Churn/"

data_en = data_path+"english/x_all.txt" #'english/out_preproc_EN.csv'
data_de = data_path +"german/x_all.txt" # 'german/out_preproc_DE.csv'

data_paths = {"en": data_en, "de": data_de}
directory = "output/"
if not os.path.exists(directory):
    os.makedirs(directory)

# Create the dataset and load the data
dataset = CrossValDataset(data_paths, train_mode, language, word_embeddings_path, max_num_words, max_seq_length)

# Prepare the data for the model
x_dict, labels_dict, y_dict, labels_to_ids = dataset.prepare_data() #embedding_matrix, vocab_size,

for lang in x_dict:
    y_pred = model.predict(x_dict[lang]).argmax(1)
    y_targ = y_dict[lang].argmax(1)

    print("len(y_pred):", len(y_pred))
    print("list(y_pred):", list(y_pred))

    print("lang =>", lang)

    _acc = accuracy_score(y_targ, y_pred)
    _f1_M = f1_score(y_targ, y_pred, average='macro')
    _recall_M = recall_score(y_targ, y_pred, average='macro')
    _precision_M = precision_score(y_targ, y_pred, average='macro')

    print("accuracy=", _acc)
    print("macro f1=", _f1_M)
    print("macro recall=", _recall_M)
    print("macro precision=", _precision_M)