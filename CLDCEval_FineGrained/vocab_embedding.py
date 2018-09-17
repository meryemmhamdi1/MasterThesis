from gensim.models import KeyedVectors
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

""" Vocabulary Creation and Convertion to IDs"""
def create_vocabulary(x_all): #, save_path
    vocab_dict = {}
    for doc in x_all:
        for token in doc:
            token = token.decode("utf-8", errors='ignore')
            # #print(token)
            if token in vocab_dict:
                vocab_dict[token] += 1
            else:
                vocab_dict[token] = 1
    vocab_list = sorted(vocab_dict)
    vocab = dict([x, y] for (y, x) in enumerate(vocab_list))

    """
    print("Saving vocabulary to pickle file")
    with open(save_path+"_vocab.p","wb") as file:
        pkl.dump(vocab, file)
    """
    return vocab, vocab_dict

""" Converting to ids """
def convert_ids(x_train_pro, x_dev_pro, x_test_pro, vocab):
    sequences_train = []
    for doc in x_train_pro:
        list_ids_sub = []
        for token in doc:
            token = token.decode("utf-8", errors='ignore')
            list_ids_sub.append(vocab[token])
        sequences_train.append(list_ids_sub)

    sequences_dev = []
    for doc in x_dev_pro:
        list_ids_sub = []
        for token in doc:
            token = token.decode("utf-8", errors='ignore')
            list_ids_sub.append(vocab[token])
        sequences_dev.append(list_ids_sub)

    sequences_test = []
    for doc in x_test_pro:
        list_ids_sub = []
        for token in doc:
            token = token.decode("utf-8", errors='ignore')
            list_ids_sub.append(vocab[token])
        sequences_test.append(list_ids_sub)

    return sequences_train, sequences_dev, sequences_test

def convert_ids_no_dev(x_train_pro, x_test_pro, vocab):
    sequences_train = []
    for doc in x_train_pro:
        list_ids_sub = []
        for token in doc:
            list_ids_sub.append(vocab[token])
        sequences_train.append(list_ids_sub)

    sequences_test = []
    for doc in x_test_pro:
        list_ids_sub = []
        for token in doc:
            list_ids_sub.append(vocab[token])
        sequences_test.append(list_ids_sub)

    return sequences_train, sequences_test


""" Padding to fixed length matrix """
def pad_fixed_length(sequences_train, sequences_dev, sequences_test, pad_value, max_sequences):
    data_train = pad_sequences(sequences_train, padding='post', maxlen=max_sequences, value=pad_value)
    data_dev = pad_sequences(sequences_dev, padding='post', maxlen=max_sequences, value=pad_value)
    data_test = pad_sequences(sequences_test, padding='post', maxlen=max_sequences, value=pad_value)
    return data_train, data_dev, data_test


def pad_fixed_length_no_dev(sequences_train, sequences_test, pad_value, max_sequences):
    data_train = pad_sequences(sequences_train, padding='post', maxlen=max_sequences, value=pad_value)
    data_test = pad_sequences(sequences_test, padding='post', maxlen=max_sequences, value=pad_value)
    return data_train, data_test

def load_gensim_model(language, model_dir, model_file, lang_dict):
    model = {}
    model_gensim = KeyedVectors.load_word2vec_format(model_dir + model_file, binary=True)
    for word in model_gensim.wv.vocab:
        model.update({lang_dict[language]+"_"+word:model_gensim[word]})
    return model

def load_fast_text(language, model_dir, model_file, lang_dict):
    with open(model_dir + model_file) as vector_file:
        word_vecs = vector_file.readlines()[1:]

    model = {}
    for word in word_vecs:
        parts = word.split(" ")
        model.update({lang_dict[language]+"_"+parts[0]: map(float, parts[1:-1])})

    return model
def load_embeddings(mode, language, model_dir, model_file, lang_dict):
    if mode == "mono":
        model = load_fast_text(language, model_dir, model_file, lang_dict)
    else:
        model = {}
        with open(model_dir + model_file) as file_model:
            data = file_model.readlines()

        print("Loading list of words and their vectors in all languages ....")
        if model_file == "joint_emb_ferreira_2016_reg-l1_mu-1e-9_epochs-50" \
                or model_file == "multi_embed_linear_projection":
            for i in tqdm(range(0, len(data))):
                lang = data[i].split(" ")[0].split("_")[1]
                if lang in ["en", "fr", "de", "it"]:
                    word = lang + "_" + data[i].split(" ")[0].split("_")[0]
                    vectors = [float(vector) for vector in data[i].split(" ")[1:]]
                    model.update({word: vectors})
        elif model_file == "semantic_spec_mrksic_2017-en_de_it_ru-ende-lang-joint-1e-09"\
                or model_file == "fasttext_en_de_fr_it.vec":
            for i in tqdm(range(0, len(data))):
                lang = data[i].split(" ")[0].split("_")[0]
                word = lang + "_" + data[i].split(" ")[0].split("_")[1]
                #if word in vocab.keys():
                vectors = [float(vector) for vector in data[i].split(" ")[1:]]
                model.update({word: vectors})
        else:
            for i in tqdm(range(0, len(data))):
                lang = data[i].split(" ")[0].split(":")[0]
                if lang in ["en", "fr", "de", "it"]:
                    word = lang + "_" + data[i].split(" ")[0].split(":")[1]
                    vectors = [float(vector) for vector in data[i].split(" ")[1:]]
                    model.update({word: vectors})

    embed_dim = len(model[list(model.keys())[0]])

    print("embed_dim=", embed_dim)
    return model, embed_dim

def build_embedding_matrix(vocab, model, embed_dim, vocab_dict):
    not_covered_words = []
    covered_words = []

    covered_count = 0
    not_covered_count = 0

    embedding_matrix = np.zeros((len(vocab) + 1, embed_dim))
    i = 0
    for word in vocab.keys():
        if word in model:
            embedding_vector = model[word]
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
            covered_words.append(word)
            covered_count += vocab_dict[word]
        else:
            not_covered_words.append(word)
            not_covered_count += vocab_dict[word]
        i += 1
    print("Number of Covered words ==>", covered_count)

    """
    print("Saving Covered Words ...")
    with open(save_path+"_covered_words.txt", "w") as file:
        for word in covered_words:
            file.write(word+"\n")

    """
    print("Number of Non-covered words ==>", not_covered_count)

    """
    print("Saving not-Covered Words ...")
    with open(save_path+"_not_covered_words.txt", "w") as file:
        for word in not_covered_words:
            file.write(word.encode("utf-8")+"\n")

    """
    return embedding_matrix
