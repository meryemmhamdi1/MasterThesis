"""
 Executes the whole pipeline (except tweet preprocessing, split by hour, and compute prior seg probabilities)
 Testing for now for one particular time window:
        Finds bursty segments
        Finds event clusters
        Evaluates found event clusters

 Options to run this code are:
        Monolingual bursty segments => Monolingual event clustering
        Monolingual bursty segments => Multilingual event clustering
        Multilingual bursty segments => Multilingual event clustering
"""
import argparse
from BurstySegmentExtraction.detect_bursty_segments import *
#from PreprocessingModule.prepare_csv_data import *
#from PreprocessingModule.preprocess_sp_tweets import *
from EventSegmentClustering.cluster_events import *
import datetime
from math import sqrt

try:
    import cPickle as pkl
except ImportError:
    import _pickle as pkl

import os
import time
import datetime
from time import strptime
import itertools
import pandas as pd


def get_args():
    """
    Command line arguments

    Arguments set the default values of command line arguments
    :return:
    """
    parser = argparse.ArgumentParser()

    """ Dataset Path Parameters """
    parser.add_argument("--data-choice", "-dc", type=str, default="/aimlx/Datasets/EvDet/world_cup_18/csv_files/",
                        help="Choice of the dataset to be used for event detection:"
                             "/aimlx/Datasets/EvDet/world_cup_14/"
                             "/aimlx/Datasets/EvDet/world_cup_18/")
    parser.add_argument("--round-choice", "-rc", type=str, default="Group Round (1st)",
                        help="Group Round (2nd), Group Round (3rd),  Round of 8, Quart-Finals, Semi-Finals, 3rd Place, Final")

    parser.add_argument("--match-choice", "-mc", type=str, default="POR-ESP")#POR-ESP")

    parser.add_argument("--start-year", "-sy", type=int, default=2018)
    parser.add_argument("--start-month", "-sm", type=int, default=6)
    parser.add_argument("--start-day", "-sd", type=int, default=28)
    parser.add_argument("--start-hour", "-sh", type=int, default=10)
    parser.add_argument("--hashtags", "-hh", type=str, default="allTags")
                        #"officialtags,teamTags1")

    parser.add_argument("--trigger-mode", "-tm", type=str, default="multi",
                        help="Other choices: multi")

    parser.add_argument("--sem-sim-mode", "-ssm", type=str, default="emb",
                        help="Other choices: tf-idf")

    #parser.add_argument("--all-lang", "-al", type=str, default="en,fr")#"ar,da,de,en,es,fa,fr,is,it,ja,ko,pl,pt,ru,sv,hr,sr")
    parser.add_argument("--event-mode", "-em", type=str, default="multi", help="Other choices: mono")

    parser.add_argument("--time-window-size", "-tws", type=int, default=2,
                        help="Size of TimeWindow")

    parser.add_argument("--sub-window-size", "-sws", type=int, default=1,
                        help="Size of SubWindow")

    parser.add_argument("--n-time-windows", "-ntw", type=int, default=1,
                        help="Root Directory of word embeddings")

    parser.add_argument("--mono-model-dir", "-momd", type=str,
                        default="/aimlx/Embeddings/MonolingualEmbeddings/",
                        help="Root Directory of multilingual word embeddings")

    parser.add_argument("--multi-model-dir", "-mmd", type=str,
                        default="/aimlx/Embeddings/MultilingualEmbeddings/",
                        help="Root Directory of monolingual word embeddings")

    parser.add_argument("--multi-model-file", "-wep", type=str,
                        #default="expert_dict_dim_red_en_de_fr_it.txt",
                        #default="multiCCA_40_normalized",
                        default="multiCCA_512_normalized",
                        help="Path of word embeddings")

    parser.add_argument("--results-dir", "-rd", type=str,
                        default="/aimlx/Results/EvDet/",
                        help="Path of where the results get saved")

    parser.add_argument("--neighbors", "-n", type=int, default=3, help="parameter in knn")

    parser.add_argument("--min-cluster-segments", "-mcs", type=int, default=10, help="parameter in knn")

    return parser.parse_args()


def union(*dicts):
    return dict(itertools.chain.from_iterable(dct.items() for dct in dicts))


if __name__ == "__main__":
    global country_langs
    country_langs = {"ARG": "es", "AUS": "en", "BEL": "fr", "BRA": "pt", "COL": "es", "CRC": "es", "CRO": "hr"
                        , "DEN": "da", "EGY": "ar", "ENG": "en", "FRA": "fr", "GER": "de", "ISL": "is", "IRN": "fa"
                        , "JPN": "ja", "KOR": "ko", "MEX": "es", "MAR": "ar", "NGA": "en","PAN": "es", "PER": "es"
                        , "POL": "pl", "POR": "pt", "RUS": "ru", "KSA": "ar", "SEN": "fr", "SRB": "sr", "ESP": "es"
                        , "SWE": "sv", "SUI": "de_fr_it", "TUN": "ar", "URU": "es"}

    st_time = time.time()
    args = get_args()
    mode = args.event_mode

    root_dir = args.data_choice
    time_window_size = args.time_window_size
    sub_window_size = args.sub_window_size
    n_time_windows = args.n_time_windows

    home_teams = []
    away_teams = []
    date_matches = []
    times_matches = []
    matches_names = []
    fixture = pd.read_csv("/aimlx/world_cup18_fixture.csv")
    fixture = fixture[fixture["Round"] == args.round_choice].reset_index()
    for i in range(len(fixture)):
        match_name = fixture.iloc[i]["home_team/code"]+"-"+fixture.iloc[i]["away_team/code"]
        if fixture.iloc[i]["to_analyze"] == 1 and match_name in ["POR-ESP"]:#"ESP-RUS", "BRA-MEX", "BEL-JPN", "COL-ENG"]: #['SWE-ENG', 'URU-FRA', 'BEL-PAN', 'FRA-PER', 'FRA-AUS', 'POR-ESP', 'ENG-PAN', 'SEN-COL', 'NGA-ARG', 'URU-POR', 'POR-ESP', 'AUS-PER', 'GER-MEX', 'BRA-CRC', 'BRA-SUI', 'ENG-BEL']:#match_name == args.match_choice:
            home_teams.append(fixture.iloc[i]["home_team/code"])
            away_teams.append(fixture.iloc[i]["away_team/code"])
            matches_names.append(fixture.iloc[i]["home_team/code"]+"-"+fixture.iloc[i]["away_team/code"])
            date_matches.append(fixture.iloc[i]["date_time"])
            times_matches.append(fixture.iloc[i]["Time"])

    for i in range(len(matches_names)):
        langs = set()
        langs.add("en")

        if "_" in country_langs[home_teams[i]]:
            for lang in country_langs[home_teams[i]].split("_"):
                langs.add(lang)
        else:
            langs.add(country_langs[home_teams[i]])

        if "_" in country_langs[away_teams[i]]:
            for lang in country_langs[away_teams[i]].split("_"):
                langs.add(lang)
        else:
            langs.add(country_langs[away_teams[i]])

        lang_set = list(langs)
        if mode == "mono":
            model_dir = args.mono_model_dir
            model_file = "wiki."
        else:
            model_dir = args.multi_model_dir
            model_file = args.multi_model_file

        # STEP 1: Check if data for that day is prepared, otherwise call DataPreparator
        # print(date_matches[i])
        start_year = date_matches[i].split("_")[2]
        start_month = date_matches[i].split("_")[1]
        start_day = date_matches[i].split("_")[0]
        start_hour = times_matches[i].split(":")[0]

        if start_day[0] == "0":
            start_day == start_day[1]
        if start_month[0] == "0":
            start_month = start_month[1]
        if start_hour[0] == "0":
            start_hour = start_hour[1]

        start = datetime.datetime(int(start_year), int(start_month), int(start_day), int(start_hour), 0, 0)
        """
        filename_dates = []
        for filename in os.listdir(root_dir + "original_tweets/"):
            date = filename.split("/")[-1].split(".")[0][3:]
            #print("filename:", root_dir + "original_tweets/"+filename)
            #print("date:", date)
            day = int(date.split("_")[0])
            month = int(date.split("_")[1])
            year = int(date.split("_")[2])
            filename_dates.append(datetime.datetime(year, month, day, args.start_hour, 0, 0))
    
        if start not in filename_dates:
            print("Preparing Dataset")
            orig_dir = "/aimlx/Datasets/EvDet/world_cup_18/"
            dp = DataPreparator(args.start_day, args.start_month, args.start_year, 0,
                                args.all_lang.split(","), orig_dir, root_dir, args.hashtags.split(","))
            dp.json_to_csv()
            
        """

        # STEP 2: Check if preprocessed, otherwise preprocess for each group of languages

        """
        original_tweets_dir = root_dir + "original_tweets/"
        cleaned_tweet_dir = root_dir + "cleaned_tweets/"
        langs = args.event_mode.split("_")[1:]
        pre_flag = [False]*len(langs)
        if not os.path.isdir(cleaned_tweet_dir):
            pre_flag = [True]*len(langs)
        else:
            for i in range(len(langs)):
                if not os.path.isfile(cleaned_tweet_dir + langs[i] + "/" + langs[i] + "_" + str(args.start_day) + "_"
                                      + str(args.start_month) + "_" + str(args.start_year)+".csv"):
                    pre_flag[i] = True
    
        for i in range(len(pre_flag)):
            if pre_flag[i]:
                print("PREPROCESSING LANGUAGE :", langs[i])
                tp = TweetPreprocessor(str(args.start_day) + "_" + str(args.start_month) + "_" + str(args.start_year), langs[i])
                print("original_tweets_dir=", original_tweets_dir)
                tp.clean_tweets(original_tweets_dir, cleaned_tweet_dir)
                
        """

        bs = BurstySegmentation()
        ed = EventDetector()

        bursty_segments = []
        sws = []
        if mode == "multi":
            for lang in lang_set:
                print("Processing lang => "+lang)
                cleaned_tweet_dir = root_dir + "cleaned_tweets/" +lang + "/"
                sub_windows_dir = cleaned_tweet_dir + "sub-windows/"
                print(" Splitting Tweets per hour for " + lang + " ...")
                print("sub_windows_dir:", sub_windows_dir)
                # STEP 3: Check if subwindows exist, otherwise create them
                #if lang == "en":
                bs.split_tweets_time_windows(cleaned_tweet_dir, start, lang, sub_windows_dir)

                # Get Bursty Segments either monolingual or multilingual
                sub_window_size = args.sub_window_size  # 1 hour
                start_time = start  # to be changed to evaluate over another time window
                n_time_windows = args.n_time_windows  #

                # Initialize the TimeWindow
                subwindows = []
                segments_subs = []
                n_subwindows = int(time_window_size/sub_window_size)
                for sub_window_no in range(n_subwindows):
                    sw = bs.read_subwindow(sub_windows_dir, start_time, sub_window_size, lang)
                    start_time += datetime.timedelta(hours=sub_window_size)
                    subwindows.append(sw)
                    segments_subs.append(sw.get_segments())

                tw = TimeWindow(subwindows)

                # Compute seg_prob for that time window
                print("Compute Seg_prob")
                seg_prob = {}
                for seg_name in tqdm(tw.get_segment_names()):
                    seg_prob[seg_name] = tw.get_segment_prob_time_window(seg_name)

                # Read seg_prob
                """
                with open(cleaned_tweet_dir+"seg_prob.p", "rb") as file:
                    seg_prob = pkl.load(file)
                """
                segments_burst_w = bs.get_bursty_segments(tw, seg_prob)
                if "mono" in args.trigger_mode:
                    # if mono then take only triggers which are bursty
                    k = int(sqrt(tw.get_total_tweet_count()))
                    bursty_segments += [(seg[0].lower(), seg[1]) for seg in sorted(segments_burst_w, key=lambda x: x[1],
                                                                                   reverse=True)[:k]]
                else:
                    # if multi then keep all triggers along with their score which will be used
                    # in the post filtering after the clustering phase
                    bursty_segments += segments_burst_w
                sws.append(segments_subs)

            sws_langs = map(list, zip(*sws))
            sws_time = []

            for sws_lang in sws_langs:
                multi_lang_segment = union(*sws_lang)
                sws_time.append(SubWindow(multi_lang_segment))

            segments = [seg[0] for seg in bursty_segments]

            print("Saving bursty segments for visualization")

            save_dir = args.results_dir + args.data_choice.split("/")[-3] + "/" + matches_names[i] + "-" + start_day \
                       + "_" + start_month + "_" + start_year + "-" + start_hour + "-" + "_".join(lang_set) + "/"
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)
            save_dir += args.sem_sim_mode + "_"
            datetime_str = str(start.year) + "-" + str('{:02d}'.format(start.month)) + "-" + \
                           str('{:02d}'.format(start.day)) + "_" + str('{:02d}'.format(start.hour))
            with open(save_dir + mode + "_" + "_".join(lang_set) + "_" + args.trigger_mode
                      + "-trigger_bursty-seg_" + datetime_str + "hour.txt", "w") as file:
                for seg in bursty_segments:
                    file.write(seg[0] + ":" + str(seg[1]) + "\n")

            if args.sem_sim_mode == "emb":
                print("Load the embedding model")
                model, embed_dim = ed.load_embeddings(segments, mode, lang, lang_set, model_dir, model_file)
            else:
                model = {}

            seg_sim, new_bursty = ed.compute_similarities(bursty_segments, sub_window_size, model, sws_time, args.sem_sim_mode)

            print("Saving semantic similarities in ", cleaned_tweet_dir+"NEW_seg_sim.txt")
            with open(cleaned_tweet_dir+"NEW_seg_sim.txt", "w") as file:
                for i in range(len(seg_sim)):
                    sim = ""
                    for j in range(len(seg_sim[i])):
                        sim += str(seg_sim[i][j]) + "\t"
                    file.write(sim+"\n")

            clusters = ed.get_knn_events(new_bursty, seg_sim, args.trigger_mode, args.neighbors, args.min_cluster_segments)

            print(clusters)

            print("Saving clusters into results folder")
            with open(save_dir + mode + "_" + "_".join(lang_set) + "_" + args.trigger_mode +"-trigger_"+
                      str(args.neighbors)+"-neigh_" + str(args.min_cluster_segments)+"-min-cluster-seg.txt", "w") as file:
                "Saving details of training"
                file.write("--------------------------------------------------------------------------------------\n")
                file.write("Extracted events for Time Window starting " + str(start) + " time_window_size:" + str(args.time_window_size)
                           + " sub_window_size: " + str(args.sub_window_size) + " using KNN with parameters: \n"
                           +str(args.neighbors) + " neighbours and " + str(args.min_cluster_segments) + " min cluster segments "
                           + "\nfound "+str(len(clusters))+" event clusters done in " + str(time.time()-st_time) + " seconds\n")
                file.write("--------------------------------------------------------------------------------------\n")

                i = 1
                for cluster in clusters:
                    cluster_str = "CLUSTER " + str(i) + ":" + "\t".join(cluster) + "\n"
                    file.write(cluster_str)
                    i += 1
        else:
            save_dir = args.results_dir + args.data_choice.split("/")[-3] + "/" + matches_names[i] + "-" + start_day \
                       + "_" + start_month + "_" + start_year + "-" + start_hour + "-" + "_".join(lang_set) + "/"
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)
            save_dir += args.sem_sim_mode + "_"
            for lang in lang_set:
                print("Processing lang => "+lang)
                cleaned_tweet_dir = root_dir + "cleaned_tweets/" +lang + "/"
                sub_windows_dir = cleaned_tweet_dir + "sub-windows/"
                print(" Splitting Tweets per hour for " + lang + " ...")
                print("sub_windows_dir:", sub_windows_dir)
                # STEP 3: Check if subwindows exist, otherwise create them
                #if lang == "en":
                bs.split_tweets_time_windows(cleaned_tweet_dir, start, lang, sub_windows_dir)

                # Get Bursty Segments either monolingual or multilingual
                sub_window_size = args.sub_window_size  # 1 hour
                start_time = start  # to be changed to evaluate over another time window
                n_time_windows = args.n_time_windows  #

                # Initialize the TimeWindow
                subwindows = []
                n_subwindows = int(time_window_size/sub_window_size)
                for sub_window_no in range(n_subwindows):
                    sw = bs.read_subwindow(sub_windows_dir, start_time, sub_window_size, lang)
                    start_time += datetime.timedelta(hours=sub_window_size)
                    subwindows.append(sw)

                tw = TimeWindow(subwindows)

                # Compute seg_prob for that time window
                print("Compute Seg_prob")
                seg_prob = {}
                for seg_name in tqdm(tw.get_segment_names()):
                    seg_prob[seg_name] = tw.get_segment_prob_time_window(seg_name)

                # Read seg_prob
                """
                with open(cleaned_tweet_dir+"seg_prob.p", "rb") as file:
                    seg_prob = pkl.load(file)
                """
                segments_burst_w = bs.get_bursty_segments(tw, seg_prob)
                if "mono" in args.trigger_mode:
                    # if mono then take only triggers which are bursty
                    k = int(sqrt(tw.get_total_tweet_count()))
                    bursty_segments = [(seg[0].lower(), seg[1]) for seg in sorted(segments_burst_w, key=lambda x: x[1],
                                                                                   reverse=True)[:k]]
                else:
                    # if multi then keep all triggers along with their score which will be used
                    # in the post filtering after the clustering phase
                    bursty_segments = segments_burst_w

                segments = [seg[0] for seg in bursty_segments]

                print("Saving bursty segments for visualization")
                datetime_str = str(start.year) + "-" + str('{:02d}'.format(start.month)) + "-" + \
                               str('{:02d}'.format(start.day)) + "_" + str('{:02d}'.format(start.hour))
                with open(save_dir + mode + "_" + lang + "_" + args.trigger_mode
                          + "-trigger_bursty-seg_" + datetime_str + "hour.txt", "w") as file:
                    for seg in bursty_segments:
                        file.write(seg[0] + ":" + str(seg[1]) + "\n")

                if args.sem_sim_mode == "emb":
                    print("Load the embedding model")
                    model, embed_dim = ed.load_embeddings(segments, mode, lang, lang_set, model_dir, model_file)
                else:
                    model = {}

                seg_sim, new_bursty = ed.compute_similarities(bursty_segments, sub_window_size, model, subwindows, args.sem_sim_mode)

                clusters = ed.get_knn_events(new_bursty, seg_sim, args.trigger_mode, args.neighbors, args.min_cluster_segments)

                print(clusters)

                print("Saving clusters into results folder")
                with open(save_dir + mode + "_" + lang + "_" + args.trigger_mode +"-trigger_"+
                          str(args.neighbors)+"-neigh_" + str(args.min_cluster_segments)+"-min-cluster-seg.txt", "w") as file:
                    "Saving details of training"
                    file.write("--------------------------------------------------------------------------------------\n")
                    file.write("Extracted events for Time Window starting " + str(start) + " time_window_size:" + str(args.time_window_size)
                               + " sub_window_size: " + str(args.sub_window_size) + " using KNN with parameters: \n"
                               +str(args.neighbors) + " neighbours and " + str(args.min_cluster_segments) + " min cluster segments "
                               + "\nfound "+str(len(clusters))+" event clusters done in " + str(time.time()-st_time) + " seconds\n")
                    file.write("--------------------------------------------------------------------------------------\n")

                    i = 1
                    for cluster in clusters:
                        cluster_str = "CLUSTER " + str(i) + ":" + "\t".join(cluster) + "\n"
                        file.write(cluster_str)
                        i += 1

