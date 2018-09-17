from CLDCEval.dataModule import data_utils
from CLDCEval.dataModule.RCV import new_processor
from CLDCEval.get_args import *
from tqdm import tqdm
import random

if __name__ == '__main__':

    global args, lang_list, lang_dict, model_lang_mapping, train_lang, test_lang_list

    lang_dict = {}
    with open("../iso_lang_abbr.txt") as iso_lang:
        for line in iso_lang:
            lang_dict.update({line.split(":")[1][:-1]: line.split(":")[0]})

    args = get_args()

    """ 2. Data Extraction """
    x_train_dict = {}
    y_train_dict = {}
    x_dev_dict = {}
    y_dev_dict = {}
    x_test_dict = {}
    y_test_dict = {}
    data_util_dict = {}

    model_dir = args.model_dir
    model_file = args.multi_model_file
    for language in args.languages.split(','):
        print("Processing language=> ", language)
        data_util = data_utils.DataUtils(args.data_rcv+"ECGA/", args.pre_dir, args.stop_pos_path, args.lemma_use,
                                         args.stop_use, language, model_dir, model_file, args.embed_dim)

        dp = new_processor.MonoProcessor(data_util, lang_dict)

        x_train_dict.update({language: dp.x_train_pro})
        y_train_dict.update({language: dp.y_train})
        x_dev_dict.update({language: dp.x_dev_pro})
        y_dev_dict.update({language: dp.y_dev})
        x_test_dict.update({language: dp.x_test_pro})
        y_test_dict.update({language: dp.y_test})
        data_util_dict.update({language: data_util})

    """ 3. Class Statistics per language """
    min_class_train = {"0":1000000, "1":1000000, "2":1000000, "3":1000000}
    min_class_dev = {"0":1000000, "1":1000000, "2":1000000, "3":1000000}
    min_class_test = {"0":1000000, "1":1000000, "2":1000000, "3":1000000}
    x_train_class_dict = {}
    x_dev_class_dict = {}
    x_test_class_dict = {}
    with open("class_statistics.txt", "w") as file:
        for language in args.languages.split(','):
            x_train_class = {}
            x_dev_class = {}
            x_test_class = {}
            print ("For language: ", language)
            for i in range(0, len(x_train_dict[language])):
                x = x_train_dict[language][i]
                y = y_train_dict[language][i]
                if y in x_train_class:
                    x_train_list = x_train_class[y]
                    x_train_list.append(x)
                    x_train_class.update({y:x_train_list})
                else:
                    x_train_class.update({y:[x]})

            for i in range(0, len(x_dev_dict[language])):
                x = x_dev_dict[language][i]
                y = y_dev_dict[language][i]
                if y in x_dev_class:
                    x_dev_list = x_dev_class[y]
                    x_dev_list.append(x)
                    x_dev_class.update({y:x_dev_list})
                else:
                    x_dev_class.update({y:[x]})

            for i in range(0, len(x_test_dict[language])):
                x = x_test_dict[language][i]
                y = y_test_dict[language][i]
                if y in x_test_class:
                    x_test_list = x_test_class[y]
                    x_test_list.append(x)
                    x_test_class.update({y:x_test_list})
                else:
                    x_test_class.update({y:[x]})

            file.write("Language "+language+"\n")
            for y in x_train_class:
                file.write("Class "+y+ " has "+ str(len(x_train_class[y])) + " training instances \n")
                file.write("Class "+y+ " has "+ str(len(x_dev_class[y])) + " validation instances \n")
                file.write("Class "+y+ " has "+ str(len(x_test_class[y])) + " testing instances \n")
                if len(x_train_class[y]) < min_class_train[y]:
                    min_class_train[y] = len(x_train_class[y])

                if len(x_dev_class[y]) < min_class_dev[y]:
                    min_class_dev[y] = len(x_dev_class[y])

                if len(x_test_class[y]) < min_class_test[y]:
                    min_class_test[y] = len(x_test_class[y])

            x_train_class_dict.update({language:x_train_class})
            x_dev_class_dict.update({language:x_dev_class})
            x_test_class_dict.update({language:x_test_class})

        for y in min_class_train:
            file.write("Minimum number of training instances for class "+y+ " is: "+str(min_class_train[y]))
            file.write("Minimum number of validation instances for class "+y+ " is: "+str(min_class_dev[y]))
            file.write("Minimum number of testing instances for class "+y+ " is: "+str(min_class_test[y]))

    print("Sampling data ...")
    max_sequences = 0
    x_train_dict = {}
    y_train_dict = {}
    x_dev_dict = {}
    y_dev_dict = {}
    x_test_dict = {}
    y_test_dict = {}
    for language in args.languages.split(','):
        x_train_list = []
        y_train_list = []
        for y in min_class_train:
            x_train_list.append(random.sample(x_train_class_dict[language][y], min_class_train[y]))
            y_train_list.append(min_class_train[y]*[y])

        x_train_list_fl = [item for sublist in x_train_list for item in sublist]
        y_train_list_fl = [item for sublist in y_train_list for item in sublist]

        x_dev_list = []
        y_dev_list = []
        for y in min_class_dev:
            x_dev_list.append(random.sample(x_dev_class_dict[language][y], min_class_dev[y]))
            y_dev_list.append(min_class_dev[y]*[y])

        x_dev_list_fl = [item for sublist in x_dev_list for item in sublist]
        y_dev_list_fl = [item for sublist in y_dev_list for item in sublist]

        x_test_list = []
        y_test_list = []
        for y in min_class_test:
            x_test_list.append(random.sample(x_test_class_dict[language][y], min_class_test[y]))
            y_test_list.append(min_class_test[y]*[y])

        x_test_list_fl = [item for sublist in x_test_list for item in sublist]
        y_test_list_fl = [item for sublist in y_test_list for item in sublist]

        x_train_dict.update({language:x_train_list_fl})
        y_train_dict.update({language:y_train_list_fl})
        x_dev_dict.update({language:x_dev_list_fl})
        y_dev_dict.update({language:y_dev_list_fl})
        x_test_dict.update({language:x_test_list_fl})
        y_test_dict.update({language:y_test_list_fl})


    for language in args.languages.split(','):
        print("language:", language)
        with open(args.data_rcv+"ECGA/"+language+"/train_sample.txt", "a") as file:
            for i in range(0,len(x_train_dict[language])):
                if len(x_train_dict[language][i]) > max_sequences:
                    max_sequences = len(x_train_dict[language][i])
                doc = " ".join(x_train_dict[language][i])
                file.write(doc.encode("utf-8")+"\t"+y_train_dict[language][i]+"\n")

        with open(args.data_rcv+"ECGA/"+language+"/dev_sample.txt", "a") as file:
            for i in range(0,len(x_dev_dict[language])):
                if len(x_dev_dict[language][i]) > max_sequences:
                    max_sequences = len(x_dev_dict[language][i])
                doc =  " ".join(x_dev_dict[language][i])
                file.write(doc.encode("utf-8")+"\t"+y_dev_dict[language][i]+"\n")

        with open(args.data_rcv+"ECGA/"+language+"/test_sample.txt", "a") as file:
            for i in range(0,len(x_test_dict[language])):
                if len(x_test_dict[language][i]) > max_sequences:
                    max_sequences = len(x_test_dict[language][i])
                doc = " ".join(x_test_dict[language][i])
                file.write(doc.encode("utf-8")+"\t"+y_test_dict[language][i]+"\n")


    print("max_sequences=",max_sequences)


