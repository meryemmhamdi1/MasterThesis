"""

Reads preprocessed documents from rcv for each language and puts them into separate files and folders

"""

import cPickle as pkl
from tqdm import tqdm
import codecs
import os


def save_labels_folders(save_path):
    for lang in ['english', 'german', 'italian', 'french']:
        if not os.path.isdir(save_path+lang+"/"):
            os.makedirs(save_path+lang+"/")
        data_full = {}
        data_full[0] = data_full[1] = data_full[2] = data_full[3] = []
        for split in ["train", "dev", "test"]:
            data = {}
            data[0] = data[1] = data[2] = data[3] = []
            print("Loading x_processed ... ")
            with open(rcv_path + "/" + lang + "/processed/" + split + "/X_"+split+"_new_"+lang+".p", "rb") as x_file:
                x_list = pkl.load(x_file)

            print("Loading labels ... ")
            with open(rcv_path + "/" + lang + "/processed/" + split + "/label_ids_"+split+"_new_"+lang+".p", "rb") as labels_file:
                labels = pkl.load(labels_file)

            print("Constructing dictionary of text documents according to their labels ... ")
            for i in tqdm(range(0, len(x_list))):
                data[labels[i]].append(" ".join(x_list[i]))
                data_full[labels[i]].append(" ".join(x_list[i]))

            for label in data:
                sub_path = save_path+lang+"/"+str(label) + "/"
                if not os.path.isdir(sub_path):
                    os.makedirs(sub_path)
                print("Saving for lang =>  ", lang, " split => ", split, " label => ", label)
                with codecs.open(sub_path + split + ".txt", "w", "utf-8") as text_file:
                    for j in tqdm(range(0,len(data[label]))):
                        text_file.write(data[label][j] + "\n")

        for label in data:
            print("Saving for lang =>  ", lang, " split => full,  label => ", label)
            with codecs.open(save_path+lang+"/"+str(label) + "/full.txt", "w", "utf-8") as text_file:
                for j in tqdm(range(0,len(data_full[label]))):
                    text_file.write(data_full[label][j] + "\n")


def save_labels_with_text():
    for lang in ['english', 'german', 'italian', 'french']:
        if not os.path.isdir(save_path+lang+"/"):
            os.makedirs(save_path+lang+"/")
        for split in ["train", "dev", "test"]:
            data = []
            print("Loading x_processed ... ")
            with open(rcv_path + "/" + lang + "/processed/" + split + "/X_"+split+"_new_"+lang+".p", "rb") as x_file:
                x_list = pkl.load(x_file)

            print("Loading labels ... ")
            with open(rcv_path + "/" + lang + "/processed/" + split + "/label_ids_"+split+"_new_"+lang+".p", "rb") as labels_file:
                labels = pkl.load(labels_file)

            print("Constructing dictionary of text documents according to their labels ... ")
            for i in tqdm(range(0, len(x_list))):
                data.append(" ".join(x_list[i])+"\t" + labels[i] + "\n")

            sub_path = save_path + lang + "/"
            if not os.path.isdir(sub_path):
                os.makedirs(sub_path)
            print("Saving for lang =>  ", lang, " split => ", split)
            with codecs.open(sub_path + split + ".txt", "w", "utf-8") as text_file:
                for j in tqdm(range(0, len(data))):
                    text_file.write(data[j])

if __name__ == '__main__':
    rcv_path = "/aimlx/RCV1_RCV2"
    save_path = "/aimlx/RCV1_RCV2/ECGA/"

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    save_labels_with_text()

