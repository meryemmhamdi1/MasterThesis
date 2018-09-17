"""
    Input: bursty segments for a particular time window
    Output: events and their trigger keywords
"""
from __future__ import division
import sys
sys.path.insert(0, '..')
from main import *
import sklearn.metrics.pairwise as sk
import numpy as np
import networkx as nx
from tqdm import tqdm
import scipy.spatial.distance as cs
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf_sim(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([text1, text2])
    return ((tfidf * tfidf.T).A) [0,1]


class EventDetector():
    #def __init__(self, timewindow):
    #    self.timewindow = timewindow

    def load_fast_text(self, language, model_dir, bursty_segments):
        print("Loading fast text")
        with open(model_dir + "wiki."+language+".vec") as vector_file:
            word_vecs = vector_file.readlines()[1:]

        model = {}
        for word in tqdm(word_vecs):
            parts = word.split(" ")
            if language + "_" + parts[0] in bursty_segments:
                model.update({language+"_"+parts[0]: map(float, parts[1:-1])})

        print("len(model): ", len(model))
        return model

    def load_embeddings(self, bursty_segments, mode, language, lang_set, model_dir, model_file):
        if "mono" in mode:
            model = self.load_fast_text(language, model_dir, bursty_segments)
        else:
            model = {}
            with open(model_dir + model_file) as file_model:
                data = file_model.readlines()

            print("Loading list of words and their vectors in all languages ....")
            if model_file == "joint_emb_ferreira_2016_reg-l1_mu-1e-9_epochs-50" \
                    or model_file == "multi_embed_linear_projection":
                for i in tqdm(range(0, len(data))):
                    lang = data[i].split(" ")[0].split("_")[1]
                    if lang in lang_set:#["en", "fr", "de", "it"]:
                        word = lang + "_" + data[i].split(" ")[0].split("_")[0]
                        vectors = [float(vector) for vector in data[i].split(" ")[1:]]
                        model.update({word: vectors})
            elif model_file == "semantic_spec_mrksic_2017-en_de_it_ru-ende-lang-joint-1e-09" \
                    or model_file == "fasttext_en_de_fr_it.vec" or model_file == "unsupervised_fastext.txt" \
                    or model_file == "supervised_fastext.txt" or model_file == "expert_dict_dim_red_en_de_fr_it.txt":
                for i in tqdm(range(0, len(data))):
                    # parts = data[i].split(" ")
                    # word = parts[0]
                    # model.update({word: map(float, parts[1:-1])})
                    lang = data[i].split(" ")[0].split("_")[0]
                    word = lang + "_" + data[i].split(" ")[0].split("_")[1]
                    if word in bursty_segments:
                        vectors = [float(vector) for vector in data[i].split(" ")[1:-1]]
                        model.update({word: vectors})
            else:
                for i in tqdm(range(0, len(data))):
                    lang = data[i].split(" ")[0].split(":")[0]
                    if lang in lang_set: #["en", "fr", "de", "it"]:
                        word = lang + "_" + data[i].split(" ")[0].split(":")[1]
                        vectors = [float(vector) for vector in data[i].split(" ")[1:]]
                        model.update({word: vectors})

        if len(model) == 0:
            print("NO BURSTY SEGMENT FOUND IN THE MODEL !!!!")
            if mode == "multi":
                embed_dim = 512
            else:
                embed_dim = 300
        else:
            embed_dim = len(model[list(model.keys())[0]])

        print("embed_dim=", embed_dim)
        return model, embed_dim

    def get_word2vec_similarity_segments(self, seg1, seg2, model): # TODO: SIMILARITY OF SEGMENTS BASED ON TWEETS IN WHICH THEY APPEARED
        if seg1 in model and seg2 in model:
            #x = np.array(model.get(seg1)).reshape(1, -1)
            #y = np.array(model.get(seg2)).reshape(1, -1)

            similarity = 1 - cs.cosine(model.get(seg1), model.get(seg2))  #sk.cosine_similarity(x, y)[0][0]

        else:
            similarity = 0

        return similarity

    def get_tf_idf_similarity_segments(self, seg1, seg2, sub_window_size, sws_time):
        s1_freq = 0
        s2_freq = 0

        similarity = 0
        #print("seg1:", seg1)
        #print("seg2:", seg2)
        for i in range(sub_window_size):
            sw = sws_time[i]
            #print("Segments:", sw.get_segment_names())

            s1 = sw.get_tweets_containing_segments(seg1)
            s2 = sw.get_tweets_containing_segments(seg2)

            if s1 is not None:
                s1_freq += len(s1)

            if s2 is not None:
                s2_freq += len(s2)

            if s1 is None or s2 is None:
                continue

            text1 = ""
            for tweet in s1:
                text1 += tweet.text + " "

            text2 = ""
            for tweet in s2:
                text2 += tweet.text + " "

            similarity += len(s1) * len(s2) * tf_idf_sim(text1, text2)

        similarity = similarity/(s1_freq * s2_freq)
        return similarity

    def compute_similarities(self, bursty_segments, sub_window_size, model, sws_time, mode):
        """
        Detects the events based on the bursty segments within the time windows
        :param bursty_segments:
        :return:
        """
        # Computing the similarity between bursty segments
        seg_sim = {}
        """
        new_bursty = set()
        n = len(bursty_segments)
        for i in range(0, n):
            seg_name = bursty_segments[i][0]
            if seg_name in model:
                new_bursty.add(bursty_segments[i])
        new_bursty = list(new_bursty)
        """
        n = len(bursty_segments)
        print("n:", n)
        for i in tqdm(range(n)):
            seg1_name = bursty_segments[i][0]
            for j in range(i, n):
                seg2_name = bursty_segments[j][0]
                if i not in seg_sim:
                    seg_sim[i] = {}
                if j not in seg_sim:
                    seg_sim[j] = {}

                if mode == "emb":
                    sim = self.get_word2vec_similarity_segments(seg1_name, seg2_name, model)
                else:
                    sim = self.get_tf_idf_similarity_segments(seg1_name, seg2_name, sub_window_size, sws_time)
                seg_sim[i][j] = sim
                seg_sim[j][i] = sim

        return seg_sim, bursty_segments

    def get_knn_events(self, bursty_segments, seg_sim, trigger_mode, neighbors, min_cluster_segments):
        n = len(bursty_segments)
        G = nx.Graph()
        G.add_nodes_from(range(n))
        k_neighbors = {}
        for i in range(n):
            k_neighbors[i] = self.get_k_neighbors(neighbors, i, seg_sim)

        for i in range(n):
            for j in range(i+1, n):
                if i in k_neighbors[j] and j in k_neighbors[i]:
                    G.add_edge(i, j)

        clusters = []
        for comp in nx.connected_components(G):
            if len(comp) >= min_cluster_segments:
                clusters.append([bursty_segments[i][0] for i in comp])

        return clusters

    def get_k_neighbors(self, k, seg, seg_sim):
        neighbor_list = []
        sim_list = [] # sim[i] = similarity of seg with neighbors[i]
        for i in seg_sim:
            if i == seg: continue
            neighbor_list.append(i)
            sim_list.append(seg_sim[seg][i])
        return [x for _,x in sorted(zip(sim_list, neighbor_list), reverse=True)][:k]

    def post_filtering(self, bursty_segments, clusters):

        return clusters

