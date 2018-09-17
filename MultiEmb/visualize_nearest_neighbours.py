import io
import numpy as np
from sklearn.decomposition import PCA
import cPickle as pkl
import matplotlib.pyplot as plt


"""
print("Loading SRC de_en embeddings to file:")
with open("/aimlx/Results/TranslationMatrices/de_en_SRC.p", "rb") as file:
    de_en_src = pkl.load(file)

print("Loading TRG de_en embeddings to file:")
with open("/aimlx/Results/TranslationMatrices/de_en_TRG.p", "rb") as file:
    de_en_trg = pkl.load(file)

"""
###########
print("Loading SRC fr_en embeddings to file:")
with open("/aimlx/Results/TranslationMatrices/fr_en_SRC.p", "rb") as file:
    fr_en_src = pkl.load(file)

print("Loading TRG fr_en embeddings to file:")
with open("/aimlx/Results/TranslationMatrices/fr_en_TRG.p", "rb") as file:
    fr_en_trg = pkl.load(file)


########
print("Loading SRC it_en embeddings to file:")
with open("/aimlx/Results/TranslationMatrices/it_en_SRC.p", "rb") as file:
    it_en_src = pkl.load(file)

print("Loading TRG it_en embeddings to file:")
with open("/aimlx/Results/TranslationMatrices/it_en_TRG.p", "rb") as file:
    it_en_trg = pkl.load(file)


def intersection(lst1, lst2, lst3):
    lst4_indices = []
    lst4_words = []
    lst4_vecs = []
    for i, value in enumerate(lst1):
        if value in lst2 and value in lst3:
            lst4_indices.append(i)
            lst4_words.append(value[0])
            lst4_vecs.append(value[1])
    return lst4_indices, lst4_words, lst4_vecs

def intersection_2(lst1, lst2):
    lst4_indices = []
    lst4_words = []
    lst4_vecs = []
    lst_2_list = [word for word, vec in lst2]
    for i, value in enumerate(lst1):
        if value[0] in lst_2_list:
            lst4_indices.append(i)
            lst4_words.append(value[0])
            lst4_vecs.append(value[1])
    return lst4_indices, lst4_words, lst4_vecs


#de_emb_words = []
fr_emb_words = []
it_emb_words = []

###
#de_emb_vecs = []
fr_emb_vecs = []
it_emb_vecs = []
en_emb_indices, en_emb_words, en_emb_vecs = intersection_2(fr_en_trg, it_en_trg) #intersection(de_en_trg, fr_en_trg, it_en_trg)
for ind in en_emb_indices[:10]:
    #de_emb_words.append(de_en_trg[ind][0])
    fr_emb_words.append(fr_en_trg[ind][0])
    it_emb_words.append(it_en_trg[ind][0])

    #de_emb_vecs.append(de_en_trg[ind][1])
    fr_emb_vecs.append(fr_en_trg[ind][1])
    it_emb_vecs.append(it_en_trg[ind][1])


pca = PCA(n_components=2, whiten=True)  # TSNE(n_components=2, n_iter=3000, verbose=2)
pca.fit(np.vstack([en_emb_vecs, fr_emb_vecs, it_emb_vecs]))
print('Variance explained: %.2f' % pca.explained_variance_ratio_.sum())

def plot_similar_word(fr_words, fr_emb, it_words, it_emb, tgt_words, tgt_emb, pca):#(de_words, de_emb, fr_words, fr_emb, it_words, it_emb, tgt_words, tgt_emb, pca):

    Y = []
    word_labels = []
    """
    for i, sw in enumerate(de_words):
        Y.append(de_emb[i])
        word_labels.append(sw)

    """
    for i, sw in enumerate(fr_words):
        Y.append(fr_emb[i])
        word_labels.append(sw)
    for i, sw in enumerate(it_words):
        Y.append(it_emb[i])
        word_labels.append(sw)

    for i, tw in enumerate(tgt_words):
        Y.append(tgt_emb[i])
        word_labels.append(tw)

    # find tsne coords for 2 dimensions
    Y = pca.transform(Y)
    x_coords = Y[:, 0]
    y_coords = Y[:, 1]

    # display scatter plot
    plt.figure(figsize=(10, 8), dpi=80)
    plt.scatter(x_coords, y_coords, marker='x')

    for k, (label, x, y) in enumerate(zip(word_labels, x_coords, y_coords)):
        color = 'blue' if k < len(fr_words) else 'red'  # src words in blue / tgt words in red
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points', fontsize=19,
                     color=color, weight='bold')

    plt.xlim(x_coords.min() - 0.2, x_coords.max() + 0.2)
    plt.ylim(y_coords.min() - 0.2, y_coords.max() + 0.2)
    plt.title('Visualization of the multilingual word embedding space')

    plt.show()

plot_similar_word(fr_emb_words, fr_emb_vecs, it_emb_words, it_emb_vecs, en_emb_words, en_emb_vecs, pca)
plt.savefig('vizembeddings.png')