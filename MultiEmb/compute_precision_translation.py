from __future__ import division
import io
import numpy as np
from tqdm import tqdm
import cPickle as pkl
import os

emb_dir = "/aimlx/Embeddings/MultilingualEmbeddings/"


def load_vec(emb_path, src_lang, trg_lang, nmax=10000000):
    vectors_src = []
    word2id_src = {}
    vectors_trg = []
    word2id_trg = {}
    with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        next(f)
        for i, line in enumerate(tqdm(f)):
                word, vect = line.rstrip().split(' ', 1)
                lang, word = word.split(":", 1)
                vect = np.fromstring(vect, sep=' ')
                if lang == src_lang:
                    vectors_src.append(vect)
                    word2id_src[word] = len(word2id_src)
                elif lang == trg_lang:
                    vectors_trg.append(vect)
                    word2id_trg[word] = len(word2id_trg)
                if len(word2id_src)+len(word2id_trg) == nmax:
                    break

    id2word_src = {v: k for k, v in word2id_src.items()}
    embeddings_src = np.vstack(vectors_src)

    id2word_trg = {v: k for k, v in word2id_trg.items()}
    embeddings_trg = np.vstack(vectors_trg)

    print("len(word2id_src):", len(word2id_src))
    print("len(word2id_trg):", len(word2id_trg))

    return embeddings_src, id2word_src, word2id_src, embeddings_trg, id2word_trg, word2id_trg


def get_nn(word, src_emb, src_id2word, tgt_emb, tgt_id2word, K=5):
    #print("Nearest neighbors of \"%s\":" % word)
    word2id = {v: k for k, v in src_id2word.items()}
    word_emb = src_emb[word2id[word]]
    scores = (tgt_emb / np.linalg.norm(tgt_emb, 2, 1)[:, None]).dot(word_emb / np.linalg.norm(word_emb))
    k_best = scores.argsort()[-K:][::-1]
    k_best_words = []
    for i, idx in enumerate(k_best):
        #print('%.4f - %s' % (scores[idx], tgt_id2word[idx]))
        k_best_words.append(tgt_id2word[idx])
    return k_best_words


dir_path = "/aimlx/dictionaries/"


def read_dict(src_lang, trg_lang):
    src_words = []
    trg_words = []
    path = dir_path + src_lang + "-" + trg_lang + "/" + src_lang + "-" + trg_lang + ".5000-6500.txt"
    with open(path) as file_:
        lines = file_.readlines()
        for line in lines:
            line = line.strip("\n")
            src_word, trg_word = line.split(" ")[0], line.split(" ")[1]
            src_words.append(src_word)
            trg_words.append(trg_word)

    return src_words, trg_words


model_file = "multiCCA_512_normalized" #"expert_dict_dim_red_en_de_fr_it.txt"

src_lang = "fr"
trg_lang = "en"
embeddings_src, id2word_src, word2id_src, embeddings_trg, id2word_trg, word2id_trg = load_vec(emb_dir+model_file, src_lang, trg_lang)

print("Computing Precision for ", src_lang, " -> ", trg_lang)
src_words, trg_words = read_dict(src_lang, trg_lang)
prec_1 = 0
prec_5 = 0

good_src_embeddings = []
good_trg_embeddings = []

for i in tqdm(range(1500)):
    if src_words[i] in list(id2word_src.values()):
        k_best_words = get_nn(src_words[i], embeddings_src, id2word_src, embeddings_trg, id2word_trg, K=5)
        if trg_words[i] == k_best_words[0]:
            prec_1 += 1
            good_src_embeddings.append((src_words[i], embeddings_src[word2id_src[src_words[i]]]))
            good_trg_embeddings.append((trg_words[i], embeddings_trg[word2id_trg[trg_words[i]]]))
        if trg_words[i] in k_best_words:
            prec_5 += 1

prec_1 = prec_1/1500
prec_5 = prec_5/1500

print(src_lang, " ->", trg_lang, " ==> prec_1",  prec_1, "prec_5", prec_5)

root_dir = "/aimlx/Results/TranslationMatrices/"+model_file.split(".")[0] + "/"
directory = os.path.dirname(root_dir)
if not os.path.isdir(directory):
    os.makedirs(directory)

print("Saving precision results to file:")
with open(root_dir + src_lang + "_"+trg_lang+".txt", "w") as file:
    file.write("prec_1:" + str(prec_1) + "prec_5:" +str(prec_5))

"""
print("Saving SRC good embeddings to file:")
with open("root_dir + src_lang + "_"+trg_lang+"_SRC.p", "wb") as file:
    pkl.dump(good_src_embeddings, file)

print("Saving TRG good embeddings to file:")
with open(root_dir + src_lang + "_"+trg_lang+"_TRG.p", "wb") as file:
    pkl.dump(good_trg_embeddings, file)

"""
"""
#lines = f.readlines()[2000000:]
            #for i, line in tqdm(enumerate(lines)):
            #if (i >= 0 and i < 10000) or (i >= 3000000 and i < 3010000):
"""