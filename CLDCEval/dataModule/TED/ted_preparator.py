import glob
import shutil
import os
import cPickle as pkl
import argparse

def get_args():
    """
    Get command line arguments.

    Arguments set the default values of command line arguments.
    """
    parser = argparse.ArgumentParser()

    """Dataset Path Parameters"""
    parser.add_argument("--start-docs", "-sc", type=int, default=0,
                        help='') ####----- NOT YET USED
    parser.add_argument("--end-docs", "-ec", type=int, default=1000,
                        help='') ####----- NOT YET USED

    return parser.parse_args()

def prepare_dataset(data_dir):
    args = get_args()
    ## Gathering the dataset by Train and Test
    language_pairs = []
    for name in glob.glob(data_dir + "*"):
        #print(name.split("/")[-1])
        language_pairs.append(name.split("/")[-1])

    # splits = ['train','test']
    #
    # doc_classes = []
    # language_dir = data_dir + "Raw/" + language_pairs[0] + "/train/*"
    # #print("language_dir:",language_dir)
    # for class_ in glob.glob(language_dir):
    #     doc_classes.append(class_.split("/")[-1])

    #sentiments = ['negative', 'positive']

    # print("doc_classes: ",doc_classes)


    # for language in language_pairs:
    #     #print ("Preparing")
    #     for split in splits:
    #         #print("Splits")
    #         for topic in doc_classes:
    #             # Gather both sentiments in one directory
    #             for sentiment in sentiments:
    #                 print("Processing => language = "+ language + " split = "+split + "topic = "+ topic +
    #                       " sentiment = " + sentiment )
    #                 print (data_dir + "Raw/" +language + "/"+ split
    #                        + "/" + topic + "/" + sentiment)
    #                 for name in glob.glob(data_dir + "Raw/" +language + "/"+ split
    #                                               + "/" + topic + "/" + sentiment+ "/*"):
    #                     print ("Moving file from => "+ name + " to => "+data_dir + "Raw/" +language + "/"+ split
    #                            + "/" + topic + "/")
    #                     shutil.move(name, data_dir + "Raw/" +language + "/"+ split
    #                                 + "/" + topic + "/")
    #
    #                 ## Delete sentiment folders
    #                 shutil.rmtree(data_dir + "Raw/" +language + "/"+ split
    #                               + "/" + topic + "/" + sentiment+ "/", ignore_errors=True)

    # for language in language_pairs:
    #     for split in splits:
    #         for topic in doc_classes:
    #             for name in glob.glob(data_dir + "Raw/" +language + "/"+ split + "/" + topic + "/*"):
    #                 # Rename all files using topic class
    #                 new_name = data_dir + "Raw/" +language + "/"+ split + "/" + topic + "/"+topic+"_"+name.split('/')[-1]
    #                 os.rename(name, new_name)
    #
    #                 shutil.move(new_name, data_dir + "Raw/" +language + "/"+ split + "/")
    #
    #             ## Delete topic folders
    #             shutil.rmtree(data_dir + "Raw/" +language + "/"+ split
    #                             + "/" + topic + "/", ignore_errors=True)

    # for language in language_pairs:
    #     lang_1 = language.split("-")[0]
    #     if not os.path.isdir(data_dir+"Raw/"+lang_1):
    #         os.makedirs(data_dir+"Raw/"+lang_1)
    #     lang_2 = language.split("-")[1]
    #     if not os.path.isdir(data_dir+"Raw/"+lang_2):
    #         os.makedirs(data_dir+"Raw/"+lang_2)

    ## Combining the splits
    # for language in language_pairs:
    #     for split in splits:
    #         for name in glob.glob(data_dir + "Raw/" +language + "/"+ split + "/*"):
    #             print("Processing file name => "+name)
    #             new_name = data_dir + "Raw/" +language + "/"+ split + "/"+ split + "_"+name.split('/')[-1]
    #             os.rename(name, new_name)
    #
    #             shutil.move(new_name, data_dir + "Raw/" +language + "/")
    #
    #         ## Delete topic folders
    #         shutil.rmtree(data_dir + "Raw/" +language + "/"+ split + "/", ignore_errors=True)

    names = []
    splits = []
    labels = []
    for language in ["en-de"]:#,"en-de","en-fr","fr-en","en-it","it-en", "en-de"]:
        lang1 = language.split("-")[0]
        lang2 = language.split("-")[1]
        X_train_lang1, Y_train_lang1, X_test_lang1, Y_test_lang1 = [], [], [], []
        X_train_lang2, Y_train_lang2, X_test_lang2, Y_test_lang2 = [], [], [], []
        for name in glob.glob(data_dir +language  + "/*"):
            names.append(name.split('/')[-1])
            print("Processing file=> ",name)
            if name.split('/')[-1].split(".")[1]!="p":
                #print("Reading file => "+name)
                split = name.split('/')[-1].split('_')[0]
                label = name.split('/')[-1].split('_')[1]
                splits.append(split)
                labels.append(label)
                # with open(name,'r') as ted_file:
                #     lines = ted_file.readlines()
                # lang = lines[0].split(" ")[0].split('_')[1]
                # x_sub = []
                # for line in lines:
                #     words_list = line.split(" ")
                #     sent  = []
                #     for word in words_list:
                #         sent.append(word.split("_")[0])
                #     x_sub.append(sent)
                #
                # if lang == lang1:
                #     if split == "train":
                #         X_train_lang1.append(x_sub)
                #         Y_train_lang1.append(label)
                #     else:
                #         X_test_lang1.append(x_sub)
                #         Y_test_lang1.append(label)
                # else:
                #     if split == "train":
                #         X_train_lang2.append(x_sub)
                #         Y_train_lang2.append(label)
                #     else:
                #         X_test_lang2.append(x_sub)
                #         Y_test_lang2.append(label)


        print("File Names: ",names[args.start_docs:args.end_docs])

        print("Label Names: ",labels[args.start_docs:args.end_docs])

        print("splits: ",splits[args.start_docs:args.end_docs])

        # with open("names_order.txt","w") as names_file_txt:
        #     for line in names:
        #         names_file_txt.write(line+"\n")
        #
        # with open("labels.txt","w") as labels_file_txt:
        #     for line in labels:
        #         labels_file_txt.write(line+"\n")
        #
        # with open("splits.txt","w") as splits_file_txt:
        #     for line in splits:
        #         splits_file_txt.write(line+"\n")

        # Save in the directory and delete all other files

        ## Train

        dict_ = {'en':'english','de':'german'}

        # print('lang= ',lang1)
        # print("Saving X_train_lang1:",len(X_train_lang1))
        # with open("/aimlx/TED/X_train_"+dict_[lang1]+".p",'wb') as x_lang1_pkl:
        #     pkl.dump(X_train_lang1,x_lang1_pkl)
        #
        # print("Saving Y_train_lang1:",len(Y_train_lang1))
        # with open("/aimlx/TED/Y_train_"+dict_[lang1]+".p",'wb') as y_lang1_pkl:
        #     pkl.dump(Y_train_lang1,y_lang1_pkl)

        # print("Saving X_train_lang2:",len(X_train_lang2))
        # with open(data_dir +language  + "/X_train_"+dict_[lang2]+".p",'wb') as x_lang2_pkl:
        #     pkl.dump(X_train_lang2,x_lang2_pkl)
        #
        # print("Saving Y_train_lang2:",len(Y_train_lang2))
        # with open(data_dir +language  + "/Y_train_"+dict_[lang2]+".p",'wb') as y_lang2_pkl:
        #     pkl.dump(Y_train_lang2,y_lang2_pkl)

        ## Test
        # print("Saving X_test_lang1:",len(X_test_lang1))
        # with open("/aimlx/TED/X_test_"+dict_[lang1]+".p",'wb') as x_lang1_pkl:
        #     pkl.dump(X_test_lang1,x_lang1_pkl)
        #
        # print("Saving Y_test_lang1:",len(Y_test_lang1))
        # with open("/aimlx/TED/Y_test_"+dict_[lang1]+".p",'wb') as y_lang1_pkl:
        #     pkl.dump(Y_test_lang1,y_lang1_pkl)

        # print("Saving X_test_lang2:",len(X_test_lang2))
        # with open(data_dir +language  + "/X_test_"+dict_[lang2]+".p",'wb') as x_lang2_pkl:
        #     pkl.dump(X_test_lang2,x_lang2_pkl)
        #
        # print("Saving Y_test_lang2:",len(Y_test_lang2))
        # with open(data_dir +language  + "/Y_test_"+dict_[lang2]+".p",'wb') as y_lang2_pkl:
        #     pkl.dump(Y_test_lang2,y_lang2_pkl)


    # for language in language_pairs:
    #     for split in splits:
    #         for name in glob.glob(data_dir + "Raw/" +language + "/"+ split + "/*"):
    #             with open(name) as file:
    #                 first_line = file.readlines()
    #             language first_line.split(" ")[0].split("_")[1]

if __name__ == '__main__':
#class DataPreparator(object):

    #data_dir = "/Users/MeryemMhamdi/EPFL/Spring2018/Thesis/Datasets/CLDC/TED/"
    data_dir = "/aimlx/English-German/"

    prepare_dataset(data_dir)