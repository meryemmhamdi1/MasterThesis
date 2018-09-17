from fasttext import FastVector
from sent_representation import *
from offline_alignment import *
from tqdm import tqdm
import argparse
import random

SEED = 152416045
TRAIN_PERC = 0.7
def get_args():
    """
    Get command line arguments.

    Arguments set the default values of command line arguments.
    """
    parser = argparse.ArgumentParser()

    """Dataset Path Parameters"""
    parser.add_argument("--src-lang", "-sl", type=str, default="de_fr_it",
                        help='one or more languages to align to English')

    parser.add_argument("--trg-lang", "-tl", type=str, default="en")
    parser.add_argument("--dict-type", "-dt", type=str, default="expert", help="choices: expert, psdo")
    parser.add_argument("--mono-emb-path", "-mep", type=str, default="/aimlx/Embeddings/MonolingualEmbeddings/wiki.")
    parser.add_argument("--par-sent-path", "-psp", type=str, default="/Users/meryemmhamdi/Documents/rig9/ParallelCorpora"
                                                                "/Europarl/20151028.de__20151028.en")

    parser.add_argument("--multi-path", "-mp", type=str, default="/aimlx/Embeddings/MultilingualEmbeddings/")
    parser.add_argument("--dim-red", "-dr", type=bool, default=True)

    return parser.parse_args()


if __name__ == '__main__':

    """ ARGUMENTS AND OPTIONS """
    args = get_args()

    # Source Languages
    source_langs = []
    if "_" in args.src_lang:
        for lang in args.src_lang.split("_"):
            source_langs.append(lang)

    # Target Language
    trg = args.trg_lang

    # Dimensionality reduction
    if args.dim_red:
        dim_red = "dim_red_"
    else:
        dim_red = ""

    # Save Path
    save_path = args.multi_path+args.dict_type+"_dict_"+dim_red+trg+"_"+args.src_lang+".txt"

    f = open(save_path, "w")
    f.close()

    """ TARGET DICTIONARY: no need for any transformation """
    trg_dictionary = FastVector(vector_file=args.mono_emb_path+trg+'.vec')

    with open(save_path, "a") as fout:
        for word in tqdm(trg_dictionary.id2word):
            vector = " ".join(map(str, list(trg_dictionary[word])))
            fout.write(trg+"_"+word + " " + vector + "\n")

    for src in source_langs:
        print("Learning the alignment from "+src+" to " + trg)
        src_dictionary = FastVector(vector_file=args.mono_emb_path+src+'.vec')

        if args.dict_type == "psdo":
            src_words = set(src_dictionary.word2id.keys())
            trg_words = set(trg_dictionary.word2id.keys())
            overlap = list(src_words & trg_words)
            bilingual_dictionary = [(entry, entry) for entry in overlap]
        else:
            with open("/aimlx/dictionaries/"+src+"-"+trg+"/"+src+"-"+trg+".txt") as file:
                src_trg_words = file.readlines()

            # PseudoRandomized shuffle dictionary pairs into train and test
            random.seed(SEED)
            random.shuffle(src_trg_words)
            train_size = int(len(src_trg_words) * TRAIN_PERC)
            test_size = len(src_trg_words) - train_size
            print("Number of full dictionary pairs is:", len(src_trg_words))
            print("Number of train dictionary pairs is:", train_size)
            print("Number of test dictionary pairs is:", test_size)

            src_trg_words_train = src_trg_words[:train_size]
            print("Saving train dictionary pairs =>>")
            with open("/aimlx/dictionaries/"+src+"-"+trg+"/"+src+"-"+trg+"_train.txt", "w") as file:
                for pair in src_trg_words_train:
                    file.write(pair)

            src_trg_words_test = src_trg_words[train_size:]
            print("Saving test dictionary pairs =>>")
            with open("/aimlx/dictionaries/"+src+"-"+trg+"/"+src+"-"+trg+"_test.txt", "w") as file:
                for pair in src_trg_words_test:
                    file.write(pair)

            src_words = []
            trg_words = []
            for src_trg in src_trg_words_train:
                words = src_trg.split(" ")
                src_words.append(words[0])
                trg_words.append(words[1].rstrip("\n"))

            overlap = list(set(src_words) & set(trg_words))
            bilingual_dictionary = [(entry, entry) for entry in overlap]

        src_word_matrix, trg_word_matrix = make_train_matrices(src_dictionary, trg_dictionary, bilingual_dictionary)
        if args.dim_red:
            transform_src_trg = learn_trans_dim_red(src_word_matrix, trg_word_matrix)
        else:
            transform_src_trg = learn_trans(src_word_matrix, trg_word_matrix)

        src_dictionary.apply_transform(transform_src_trg)

        # Prepare the sentences
        """
        parallel_sent_path = "/Users/meryemmhamdi/Documents/rig9/ParallelCorpora/Europarl/20151028.de__20151028.en"
        sents_en = []
        sents_de = []
        with open(parallel_sent_path) as file:
            for line in file:
                parts = line.split(" ||| ")
                sents_en.append(parts[1])
                sents_de.append(parts[0])
    
        sent_rep = SentRepresentation(sents_de, sents_en, de_dictionary, en_dictionary, "word_avg")
    
        source_sent_matrix, target_sent_matrix = make_sent_train_matrices(sent_rep.sents_rep_de, sent_rep.sents_rep_en)
    
        source_matrix = np.array(source_word_matrix + source_sent_matrix)
        target_matrix = np.array(target_word_matrix + target_sent_matrix)
        """

        with open(save_path, "a") as fout:
            for word in tqdm(src_dictionary.id2word):
                vector = " ".join(map(str, list(src_dictionary[word])))
                fout.write(src+"_"+word + " " + vector + "\n")













