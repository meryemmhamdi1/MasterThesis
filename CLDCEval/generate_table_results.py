"""
This code is for creating tables that contain different results using different methodologies

Created on Mon Mar 19 2018

@author: Meryem M'hamdi (Meryem.Mhamdi@epfl.ch)

"""

import argparse
import pandas as pd
import numpy as np
import cPickle as pkl


def get_args():
    """
    Get command line arguments.

    Arguments set the default values of command line arguments.
    """
    parser = argparse.ArgumentParser()

    """Dataset Path Parameters"""
    parser.add_argument("--choice-model", "-cm", type=str, default="mlp_fine_tune",
                        help='Choice of the model:  '
                             'cnn'
                             'mlp'
                             'mlp_fine_tune'
                             'tf_idf_avgperc'
                             'svm')
    parser.add_argument("--epoch", "-ep", type=int,
                        default=1,#"/Users/MeryemMhamdi/GoogleDriveEPFL/Gdrive Thesis/4 Results/",
                        help='Number of epochs')
    parser.add_argument("--results-dir", "-rd", type=str,
                        default="/aimlx/",#"/Users/MeryemMhamdi/GoogleDriveEPFL/Gdrive Thesis/4 Results/",
                        help='Path of the root directory for results')

    return parser.parse_args()


if __name__ == "__main__":

    args = get_args()

    if args.choice_model == "mlp":
        sub_dir = "MLP_Keras_Models_RCV/"
    elif args.choice_model == "mlp_fine_tune":
        sub_dir = "MLPTUNE_Keras_Models_RCV/"
    elif args.choice_model == "cnn":
        sub_dir = "CNN_Keras_Models_RCV/"
    elif args.choice_model == "tf_idf_avgperc":
        sub_dir = "TF_IDF_AVG_Perc_RCV/"
    elif args.choice_model == "svm":
        sub_dir = "SVM_Keras_Models_RCV/"
    else:
        sub_dir = "svm/"

    models_dir = [ 'english_mono_results.p','german_mono_results.p', 'french_mono_results.p',
                   'italian_mono_results.p',  ## Mono
                   #'en_multiSkip_40_normalized_results.p',
                   # one or two languages including English: 'en+fr_en+de+fr+it', 'en+it_en+de+fr+it',
                   #'en+fr+de_en+de+fr+it', 'en+it+de_en+de+fr+it', 'en+it+fr_en+de+fr+it',
                   # three languages inlcuding English
                   'en,de,fr,it_multiSkip_40_normalized_results.p']#'en,de,fr,it_multiSkip_40_normalized_results.p']  ## all languages



    metrics_list = []

    all_lang = ["en", "de", "fr", "it"]
    lang_dict = {"english": "en", "german": "de", "french": "fr", "italian": "it"}
    table_whole = []
    for i in range(0, len(models_dir)):
        table_values = []
        print("Loading results= ", models_dir[i])
        with open(args.results_dir+sub_dir+models_dir[i]) as file:
            results_dict = pkl.load(file)

        # Test Results
        if "multi" in models_dir[i].split("_")[1]:
            test_lang = all_lang
        else:
            test_lang = [lang_dict[models_dir[i].split("_")[0]]]
        print("Calculating results")
        for lang in all_lang:
            if lang in test_lang:
                if True:
                    accuracy = results_dict['test_metrics_' + lang][args.epoch]["acc"]

                    print ("accuracy:", accuracy)

                    f1_macro = results_dict['test_metrics_' + lang][args.epoch]["f1_macro"]
                    #f1_micro = results_dict['test_metrics_' + lang][args.epoch]["f1_micro"]

                    print ("macro f1 score:", f1_macro)
                    precision_macro = results_dict['test_metrics_' + lang][args.epoch]["precision_macro"]
                    #precision_micro = results_dict['test_metrics_' + lang][args.epoch]["precision_micro"]

                    recall_macro = results_dict['test_metrics_' + lang][args.epoch]["recall_macro"]
                    #recall_micro = results_dict['test_metrics_' + lang][args.epoch]["recall_micro"]

                else:
                    accuracy = results_dict['test_accs'][args.epoch]

                    f1_macro = results_dict['test_f1s'][args.epoch]
                    #f1_micro = results_dict['test_metrics_' + lang][args.epoch]["f1_micro"]

                    precision_macro = results_dict['test_precisions'][args.epoch]
                    #precision_micro = results_dict['test_metrics_' + lang][args.epoch]["precision_micro"]

                    recall_macro = results_dict['test_recalls'][args.epoch]

            else:
                accuracy, f1_macro, f1_micro, precision_macro, precision_micro, recall_macro, recall_micro = \
                    '-', '-', '-', '-', '-', '-', '-'
            table_values.append(accuracy)
            #table_values.append(precision_micro)
            #table_values.append(recall_micro)
            #table_values.append(f1_micro)
            table_values.append(precision_macro)
            table_values.append(recall_macro)
            table_values.append(f1_macro)
        table_whole.append(table_values)

    table_values_arr = np.transpose(np.array(table_whole))

    ## 1. Loading the results from pickle files from the different pickle files in the directory
    # for name in glob.glob(args.results_dir + sub_dir + "*"):
    #     directories= []
    #     directories.append(name.split("/")[-1])



    tuples = list(zip(*[['EN', 'EN', 'EN', 'EN', 'EN','EN',
                         'DE', 'DE', 'DE', 'DE', 'DE','DE',
                         'FR', 'FR', 'FR', 'FR', 'FR', 'FR',
                         'IT', 'IT', 'IT', 'IT', 'IT', 'IT'],
                        4*['Accuracy', #'micro Precision', 'micro Recall', 'micro f1',
                           'Macro Precision', 'Macro Recall', 'micro f1']]))

    column_names = [ 'EN(mono)', 'DE(mono)', 'FR(mono)', 'IT(mono)', 'EN+DE+FR+IT(multi)'] #'EN(multi)',

    index = pd.MultiIndex.from_tuples(tuples)

    df = pd.DataFrame(table_values_arr, index=index, columns=column_names)

    df.to_csv(args.results_dir+sub_dir + "mlp_tune_res_tab_"+str(args.epoch)+".csv", sep='\t', encoding='utf-8')

