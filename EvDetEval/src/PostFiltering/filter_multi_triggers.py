"""
 Input is event clusters, takes each event triggers and computes their combined probability of appearance p_{t}
"""
import argparse
import sys
sys.path.insert(0, '..')
from BurstySegmentExtraction.detect_bursty_segments import *
from EventSegmentClustering.cluster_events import *
import datetime
from math import sqrt
from math import exp, sqrt, log10
import cPickle as pkl
import os
import time

def get_args():
    """
    Command line arguments

    Arguments set the default values of command line arguments
    :return:
    """
    parser = argparse.ArgumentParser()

    """ Dataset Path Parameters """
    parser.add_argument("--data-choice", "-dc", type=str, default="world_cup_18/",
                        help="Choice of the dataset to be used for event detection:"
                             "world_cup_14/"
                             "world_cup_18/")

    parser.add_argument("--data-path", "-dp", type=str, default="/aimlx/Datasets/EvDet/world_cup_18/csv_files/",
                        help="Path of the dataset")

    parser.add_argument("--start-year", "-sy", type=int, default=2018)
    parser.add_argument("--start-month", "-sm", type=int, default=6)
    parser.add_argument("--start-day", "-sd", type=int, default=17)
    parser.add_argument("--start-hour", "-sh", type=int, default=15)

    parser.add_argument("--neighbors", "-n", type=int, default=3, help="parameter in knn")

    parser.add_argument("--min-cluster-segments", "-mcs", type=int, default=10, help="parameter in knn")

    parser.add_argument("--trigger-mode", "-tm", type=str, default="multi",
                        help="Other choices: multi")

    parser.add_argument("--event-mode", "-em", type=str, default="multi_es_de",
                        help="Other choices: mono_en, mono_fr, mono_it, mono_de")

    parser.add_argument("--time-window-size", "-tws", type=int, default=2,
                        help="Size of TimeWindow")

    parser.add_argument("--sub-window-size", "-sws", type=int, default=1,
                        help="Size of SubWindow")

    parser.add_argument("--n-time-windows", "-ntw", type=int, default=1,
                        help="Root Directory of word embeddings")

    parser.add_argument("--results-dir", "-rd", type=str, default="/aimlx/Results/EvDet/",
                        help="Path of where the results get saved")

    return parser.parse_args()


def sigmoid(x):
    return 1/(1+exp(-x))


def get_bursty_clusters(tw, tweet_count, user_count, event_prob):
    events_burst_w = []
    for event_name in event_prob:
        prob = seg_prob.get(seg_name, -1)
        if prob == -1:
            #print('NEW EVENT CLUSTER:', event_name)
            bursty_score = log10(1+user_count[event_name])
        else:
            seg_mean = tweet_count[event_name] * prob
            seg_std_dev = sqrt(tweet_count[event_name] * prob * (1 - prob))
            bursty_score = sigmoid(10 * (tweet_count[event_name] - seg_mean - seg_std_dev)/(seg_std_dev)) \
                           * log10(1+user_count[event_name])

        events_burst_w.append((event_name, bursty_score))
    k = int(sqrt(len(events_burst_w)))  #### TUNABLE
    bursty_events = [event[0] for event in sorted(events_burst_w, key=lambda x: x[1], reverse=True)[:k]]
    return bursty_events


if __name__ == "__main__":

    st_time = time.time()
    bs = BurstySegmentation()
    ed = EventDetector()
    args = get_args()
    root_dir = args.data_path
    """  Read the event clusters """
    with open(args.results_dir + args.data_choice + args.event_mode + "/" + args.trigger_mode + "-trigger_" +
              str(args.neighbors)+"-neigh_" + str(args.min_cluster_segments)+"-min-cluster-seg.txt") as file:
        clusters = file.readlines()[5:]

    triggers = []
    for event in clusters:
        triggers.append(event.split(":")[1].split(" "))

    """ Read language combined tweets from a random TimeWindow of interest to the event clusters """
    # Time Window related variables
    time_window_size = args.time_window_size
    sub_window_size = args.sub_window_size
    n_time_windows = args.n_time_windows

    print(args.start_month)

    start = datetime.datetime(args.start_year, args.start_month, args.start_day, args.start_hour, 0, 0)

    lang_set = args.event_mode.split("_")[1:]

    # Initialize the TimeWindow
    subwindows = []
    n_subwindows = int(time_window_size/sub_window_size)
    for sub_window_no in range(n_subwindows):
        start_time = start
        segments = {}
        for lang in lang_set:
            cleaned_tweet_dir = root_dir + "cleaned_tweets/" +lang + "/"
            sub_windows_dir = cleaned_tweet_dir + "sub-windows/"
            segments.update(bs.read_subwindow_lang(sub_windows_dir, start_time, sub_window_size, lang))

            start_time += datetime.timedelta(hours=sub_window_size)
        subwindows.append(SubWindow(segments))

    tw = TimeWindow(subwindows)

    seg_prob = {}
    tweet_count = {}
    user_count = {}
    for i in tqdm(range(len(triggers))):
        user_set = set()
        for seg_name in triggers[i]:
            if "cluster_"+str(i) not in tweet_count:
                tweet_count.update({"cluster_"+str(i): 0})
            else:
                tweet_count["cluster_"+str(i)] += tw.get_segment_freq_time_window(seg_name)
            for sw in tw.subwindows:
                #print("segment names: ", sw.get_segment_names())
                segment = sw.segments.get(seg_name, None)
                if segment is not None:
                    print("NOT NONE")
                    user_set = user_set.union(segment.user_set)
        user_count["cluster_"+str(i)] = len(user_count)
        seg_prob["cluster_"+str(i)] = tweet_count["cluster_"+str(i)]/tw.get_total_tweet_count()

    """ Compute combined bursty weight of event cluster and get top K bursty event clusters """
    bursty_events = get_bursty_clusters(tw, tweet_count, user_count, seg_prob)

    """ Save bursty event clusters """
    filename = args.trigger_mode + "-trigger-POST-FILTER_" + str(args.neighbors)+"-neigh_" + \
               str(args.min_cluster_segments)+"-min-cluster-seg.txt"

    print(bursty_events)
    with open(args.results_dir + args.data_choice + args.event_mode + "/" + filename, "w") as file:
        for event in bursty_events:
            cluster_id = int(event.split("_")[1])
            #print(clusters[cluster_id])
            file.write(clusters[cluster_id])




