import argparse

def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--word-embeddings-path", type=str,
                        #default="/aimlx/Embeddings/MultilingualEmbeddings/fasttext_en_de_fr_it.vec")
                        #default="/aimlx/Embeddings/MultilingualEmbeddings/semantic_spec_mrksic_2017-en_de_it_ru-ende-lang-joint-1e-09")
                        #default="/aimlx/Embeddings/MultilingualEmbeddings/Churn_full_fastext_en_de.vec")
                        #default="/aimlx/Embeddings/MultilingualEmbeddings/full_fastext_en_de.vec")
                        #default="/aimlx/Embeddings/MultilingualEmbeddings/multilingual_embeddings_joint_en_de_fr_it.txt") ###
                        #default="/aimlx/Embeddings/MultilingualEmbeddings/sup_en-de.txt")
                        #default="/aimlx/Embeddings/MonolingualEmbeddings/wiki.en.vec"
                        #default="/aimlx/Embeddings/MonolingualEmbeddings/wiki.de.vec")
                        default="/aimlx/Embeddings/MultilingualEmbeddings/expert_dict_dim_red_de_en.txt")
    parser.add_argument("--language", type=str, default="de")
    parser.add_argument("--train-mode", type=str, default="en,de") #,de
    parser.add_argument("--network-type", type=str, default="cg_att")
    parser.add_argument("--kernel-sizes", type=list, default=[2])#default=[2, 3])
    parser.add_argument("--filters", type=int, default=256)
    parser.add_argument("--num-of-units", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--dataset", type=str, default="churn")
    parser.add_argument("--n-classes", type=int, default=2)

#"/aimlx/MonolingualEmbeddings/german.model"
#"/aimlx/MultilingualEmbeddings/multiSkip_40_normalized"
#"/aimlx/MonolingualEmbeddings/GoogleNews-vectors-negative300.bin"
    return parser.parse_args()


