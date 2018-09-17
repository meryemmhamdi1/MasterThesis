from fasttext import FastVector
import cPickle as pkl
from tqdm import tqdm

mono_dir = "/aimlx/Embeddings/MonolingualEmbeddings/"

fr_dictionary = FastVector(vector_file=mono_dir + "wiki.fr.vec")

en_dictionary = FastVector(vector_file=mono_dir + "wiki.en.vec")

de_dictionary = FastVector(vector_file=mono_dir + "wiki.de.vec")

it_dictionary = FastVector(vector_file=mono_dir + "wiki.it.vec")


## Evaluation of the quality of the embeddings
print("Applying alignments for French")
fr_dictionary.apply_transform("alignment_matrices/fr.txt")

print("Applying alignments for English")
en_dictionary.apply_transform("alignment_matrices/en.txt")

print("Applying alignments for German")
de_dictionary.apply_transform("alignment_matrices/de.txt")

print("Applying alignments for Italian")
it_dictionary.apply_transform("alignment_matrices/it.txt")

with open("/aimlx/CLDC_Results/CNN_KerasModels_RCV/en,de,fr,it_fasttext_en_de_fr_it.vec_vocab.p", "rb") as file:
    vocab = pkl.load(file)

# Saving the transformed embeddings
print("Saving the transformed embeddings")
with open("/aimlx/Embeddings/MultilingualEmbeddings/fasttext_en_de_fr_it.vec", "w") as fout:
    for word in tqdm(en_dictionary.id2word):
        if "en_"+word in vocab:
            vector = " ".join(map(str, list(en_dictionary[word])))
            fout.write("en_"+word+ " "+ vector + "\n")

    for word in tqdm(fr_dictionary.id2word):
        if "fr_"+word in vocab:
            vector = " ".join(map(str, list(fr_dictionary[word])))
            fout.write("fr_"+word+ " "+ vector + "\n")

    for word in tqdm(de_dictionary.id2word):
        if "de_"+word in vocab:
            vector = " ".join(map(str, list(de_dictionary[word])))
            fout.write("de_"+word+ " "+ vector + "\n")

    for word in tqdm(it_dictionary.id2word):
        if "fr_"+word in vocab:
            vector = " ".join(map(str, list(it_dictionary[word])))
            fout.write("it_"+word+ " "+ vector + "\n")
