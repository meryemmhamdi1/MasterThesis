import argparse

def get_args():
    """
    Get command line arguments.

    Arguments set the default values of command line arguments.
    """
    parser = argparse.ArgumentParser()

    """Dataset Path Parameters"""
    parser.add_argument("--data-choice", "-dc", type=str, default="rcv-multi-label",#"churn"
                        help='Choice of the dataset to be used for Crosslingual Document Classification: '
                             'dw for DeutscheWelle'
                             'rcv for Reuters Dataset'
                             'rcv-bal for Balanced RCV'
                             'churn for Churn Dataset'
                             'ted for TED Corpus')

    parser.add_argument("--single-label", "-sl", type=bool, default=True)
    parser.add_argument("--n-classes", "-nc", type=bool, default=55)# 55, 103

    parser.add_argument("--data-dw", "-ddw", type=str,
                        default="/aimlx/mhan/common_data/dw_general/",
                        help='The higher level directory path of the DeutscheWelle train/dev/test dataset')

    parser.add_argument("--data-ted", "-dted", type=str,
                        default="/aimlx/Datasets/TED/",
                        help='The higher level directory path of TED Corpus')

    parser.add_argument("--data-rcv", "-drcv", type=str,
                        default="/aimlx/Datasets/RCV/Preprocessed/",
                        help='The root directory path of RCV1 and RCV2 multilingual Corpus already preprocessed'
                             'and saved on ')

    parser.add_argument("--data-rcv-multi-label", "-drcvml", type=str,
                        default="/aimlx/Datasets/RCV/MultiLabels/",
                        help='The root directory path of RCV1 and RCV2 multilingual Corpus already preprocessed'
                             'and saved on ')

    parser.add_argument("--data-rcv-bal", "-drb", type=str, default="/aimlx/Datasets/RCV/Preprocessed_Balanced/",
                        help='The higher level directory path of RCV balanced split across languages')

    parser.add_argument("--data-churn", "-dchurn", type=str,
                        default="/aimlx/Datasets/ChurnDet/",
                        #default="/Users/meryemmhamdi/Documents/rig9/ChurnDataset/",
                        help='Multilingual Churn Dataset')

    parser.add_argument("--pre-dir", "-fd", type=str,
                        default="processed/",
                        help='Directory inside of each language containing train, dev and test raw and processed pickle documents')

    ## Choosing between training using Monolingual and Multilingual embeddings
    parser.add_argument("--mode", type=str, default="mono", help='Whether to train monolingual or multilingual embeddings')

    """Parameters and Files related to Embedding models and pickles files"""
    ## Monolingual Models
    parser.add_argument("--w2v-dir", "-w2dir", type=str,
                        default="/aimlx/Embeddings/MonolingualEmbeddings/",
                        help='Path of monolingual word vector models')

    parser.add_argument("--language", "-lang", type=str, default="english", help='chosen language')

    parser.add_argument("--w2v-en", "-w2en", type=str,
                        default="wiki.en.vec", #"GoogleNews-vectors-negative300.bin",
                        help='Path to the pickle file with the list of vectors for words in the input vocabulary in English')

    parser.add_argument("--w2v-de", "-w2de", type=str,
                        default="wiki.de.vec", #"german.model",
                        help='Path to the pickle file with the list of vectors for words in the input vocabulary in German')

    parser.add_argument("--w2v-fr", "-w2fr", type=str,
                        default="wiki.fr.vec",
                        help='Path to the pickle file with the list of vectors for words in the input vocabulary in French')

    parser.add_argument("--w2v-it", "-w2it", type=str,
                        default="wiki.it.vec",
                        help='Path to the pickle file with the list of vectors for words in the input vocabulary in Italian')

    ## Multilingual Models
    parser.add_argument("--model-dir", "-md", type=str,
                        default="/aimlx/Embeddings/MultilingualEmbeddings/",
                        #default="/Users/meryemmhamdi/Desktop/rig5/home/meryem/meryem/Embeddings/MultilingualEmbeddings/",
                        help='Path to multilingual word vector models')

    # multi_embed_linear_projection
    # multiSkip_40_normalized
    # multiCCA_512_normalized
    # joint_emb_ferreira_2016_reg-l1_mu-1e-9_epochs-50
    #semantic_spec_mrksic_2017-en_de_it_ru-ende-lang-joint-1e-09
    # #"multi_embed_linear_projection.txt"
    parser.add_argument("--multi-model-file", "-mmf", type=str, default="multiSkip_40_normalized",
                        help='multilingual embedding model')

    """Parameters related to Pre-processing"""
    parser.add_argument("--stop-pos-path", "-spp", type=str, default="",
                        help='Path of the file that lists the pos tags of the words that need to be removed')

    parser.add_argument("--lemma-use", "-lu", type=bool, default=False,
                        help='Whether to use lemmatization or not')

    parser.add_argument("--stop-use", "-su", type=bool, default=False,
                        help='Whether to use stopword removal or not')

    """Parameters related to training of the model"""
    parser.add_argument("--model-choice", "-mdc", type=str, default="cnn",
                        help='The choice of model: mlp, cnn, linear-svm, mlp-tuned, gru-att')

    parser.add_argument("--languages", "-langs", type=str, default="english,german,french,italian",
                        help='the list of languages separated by comma')

    parser.add_argument("--multi-train", "-mt", type=str, default="en,de,fr,it",
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

    parser.add_argument("--batch_size", "-bs", type=int, default=100, help='the size of minibatch')

    parser.add_argument("--epochs", "-ep", type=int, default=20, help='The number of epochs used to train the model')

    parser.add_argument("--filter-sizes", "-fs", type=str, default="3,4,5",
                        help="The size of each filter in the multi-filter CNN")

    parser.add_argument("--dropout", "-dp", type=float, default=0.7, help='Dropout Layer percentage')
    parser.add_argument("--dense", "-de", type=int, default=512, help='Size of Dense Layer')
    parser.add_argument("--learning-rate", "-lra", type=float, default=1e-3, help='')
    parser.add_argument("--beta-1", "-b1", type=float, default=0.7, help='')
    parser.add_argument("--beta-2", "-b2", type=float, default=0.999, help='')
    parser.add_argument("--epsilon", "-eps", type=float, default=1e-08, help='')


    parser.add_argument("--num-filters", "-nf", type=int, default=300, help="Number of feature maps per filter type")

    # GRU Related parameters
    parser.add_argument("--num-of-units", "-nu", type=int, default=50, help="Number of units for GRU")
    parser.add_argument("--bidirectional", "-bi", type=bool, default=True, help="Whether to use Bidirectional GRU or not")

    parser.add_argument("--model-save-path", "-rmp", type=str, default="/aimlx/Results/CLDC/",
                        help='The root path where the model should be saved')

    parser.add_argument("--model_file", "-mf", type=str, default="model.yaml",
                        help='The path where the model should be saved')

    parser.add_argument("--model-weights-path", "-mwp", type=str, default="model.h5",
                        help='The path where the model should be saved with its weights')

    return parser.parse_args()
