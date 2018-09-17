"""
1. Splits the tweets into categories with respect to the time window in which they appear:
        We choose a time window of three hours consisting of three sub-windows of one hour each
2. Build a data struct called Segment which keeps information relative to that:
        Tweets
"""
from __future__ import absolute_import
from __future__ import division
import pandas as pd
import os
from tqdm import tqdm
from math import exp, sqrt, log10
from .TimeWindow import *
import datetime
from .segment import *
import numpy as np
import sys
sys.path.insert(0, '..')
from main import *
try:
    from spacy.lang.en.stop_words import STOP_WORDS as en_stop_words
    from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop_words
    from spacy.lang.it.stop_words import STOP_WORDS as it_stop_words
    from spacy.lang.de.stop_words import STOP_WORDS as de_stop_words
    from spacy.lang.pt.stop_words import STOP_WORDS as pt_stop_words
    from spacy.lang.es.stop_words import STOP_WORDS as es_stop_words
except:
    from nltk.corpus import stopwords
    en_stop_words = set(stopwords.words('english'))
    fr_stop_words = set(stopwords.words('french'))
    it_stop_words = set(stopwords.words('italian'))
    de_stop_words = set(stopwords.words('german'))
    pt_stop_words = set(stopwords.words('portuguese'))
    es_stop_words = set(stopwords.words('spanish'))

try:
    import cPickle as pkl
except ImportError:
    import _pickle as pkl

def sigmoid(x):
    return 1/(1+exp(-x))

class BurstySegmentation():
    #def __init__(self):

    def split_tweets_time_windows(self, cleaned_tweet_dir, date, lang, sub_windows_dir):
        """
        Given a csv file containing tweets for a given period of time, split the files into different files based
        on the hour when it was written
        :param cleaned_tweet_dir:  directory containing the source csv file
        :param sub_windows_dir: directory to save csv files per hour
        :return:
        """
        if str(date.day)[0] == "0":
            day_str = str(date.day)[1]
        else:
            day_str = str(date.day)
        if str(date.month)[0] == "0":
            month_str = str(date.month)[1]
        else:
            month_str = str(date.month)
        date_str = day_str + "_" + month_str + "_" + str(date.year)
        date_files = {}
        if not os.path.isdir(sub_windows_dir):
            os.mkdir(sub_windows_dir)
        for dir_path, sub_dir_list, file_list in os.walk(cleaned_tweet_dir):
            for fname in file_list:
                if fname.split(".")[-1] == "csv" and fname == lang + "_" + date_str + ".csv":
                    df = pd.read_csv(dir_path + "/" + fname, sep="|")
                    df = df[pd.notnull(df['date'])]
                    for i in tqdm(range(0, len(df))):
                        if "_" in df["date"].iloc[i]:
                            date = df["date"].iloc[i].split("_")[0]
                            year = df["date"].iloc[i].split("_")[0].split("-")[0]
                            month = df["date"].iloc[i].split("_")[0].split("-")[1]
                            day = df["date"].iloc[i].split("_")[0].split("-")[2]
                            hr = df["date"].iloc[i].split("_")[1].split(":")[0]
                            min = df["date"].iloc[i].split("_")[1].split(":")[1]
                        else:
                            date = df["date"].iloc[i].split(" ")[0]
                            year = df["date"].iloc[i].split(" ")[0].split("-")[0]
                            month = df["date"].iloc[i].split(" ")[0].split("-")[1]
                            day = df["date"].iloc[i].split(" ")[0].split("-")[2]
                            hr = df["date"].iloc[i].split(" ")[1].split(":")[0]
                            min = df["date"].iloc[i].split(" ")[1].split(":")[1]

                        date_string = year + "-"
                        if month[0] == "0":
                            date_string += month[1] + "-"
                        else:
                            date_string += month + "-"

                        if day[0] == "0":
                            date_string += day[1] + "_"
                        else:
                            date_string += day + "_"

                        text = df["text"].iloc[i]
                        username = df["username"].iloc[i]
                        tweet_id = df["id"].iloc[i]
                        tweet = Tweet(tweet_id, date, hr, min, text, username)
                        if hr[0] == "0" and len(hr)==2:
                            date_hr = date_string + hr[1]
                        else:
                            date_hr = date_string + hr

                        if date_hr not in date_files:
                            date_files[date_hr] = []
                        date_files[date_hr].append(tweet)

        for date in date_files:
            new_df = pd.DataFrame()
            ids = []
            usernames = []
            dates = []
            texts = []
            for tweet in date_files[date]:
                ids.append(tweet.tweet_id)
                usernames.append(tweet.username)
                dates.append(tweet.date + "_" + tweet.hour +":" +tweet.minutes)
                texts.append(tweet.text)

            new_df["id"] = ids
            new_df["username"] = usernames
            new_df["date"] = dates
            new_df["text"] = texts

            save_path = sub_windows_dir + date + "hour.csv"
            if not os.path.isfile(save_path):
                new_df.to_csv(save_path, sep="|")

        print("Done!")

    def read_subwindow(self, root_path, start_time, sub_window_size, lang):
        stop_words = list(en_stop_words) + list(fr_stop_words) + list(it_stop_words) + list(de_stop_words) + \
                     list(pt_stop_words) + list(es_stop_words)
        print("Reading subwindow",sub_window_size)
        segments = {} # dictionary of segments: key is segment name and value is Segment class (contains info about tweets and so on)
        for _ in range(sub_window_size):# read as many sub-windows as there are and aggregate
            fname = str(start_time.year) + "-" + str(start_time.month) + "-" +\
                    str(start_time.day) + "_" + str(start_time.hour) + "hour.csv"

            #print("fname=", root_path+fname)
            if os.path.isfile(root_path+fname):
                print("Reading file =>", fname)
                df = pd.read_csv(root_path+fname, sep="|")
                df = df[pd.notnull(df['text'])]
                for j in tqdm(range(0, len(df))):
                    if df["text"].iloc[j] is not None:
                        segmentation = [lang + "_" +seg_name for seg_name in df["text"].iloc[j].split(" ")
                                        if seg_name not in stop_words]
                        for seg in segmentation:
                            if not seg in segments:
                                segments[seg] = Segment(seg)
                            tweet_id = df["id"].iloc[j]
                            date = df["date"].iloc[j].split("_")[0]
                            hour = df["date"].iloc[j].split("_")[1].split(":")[0]
                            minutes = df["date"].iloc[j].split("_")[1].split(":")[1]
                            text = df["text"].iloc[j]
                            username = df["username"].iloc[j]
                            segments[seg].add_tweet(Tweet(tweet_id, date, hour, minutes, text, username))
                print("Number of Tweets Found: ", len(df))
            #else:
            #    print("Could not find file ", root_path+fname)

            # Increase the subwindow by 1 hour
            start_time += datetime.timedelta(hours=1)

        sw = SubWindow(segments)
        return sw

    def read_subwindow_lang(self, root_path, start_time, sub_window_size, lang):
        segments = {} # dictionary of segments: key is segment name and value is Segment class (contains info about tweets and so on)
        for _ in range(sub_window_size):# read as many sub-windows as there are and aggregate
            fname = str(start_time.year) + "-" + str(start_time.month) + "-" + \
                    str(start_time.day) + "_" + str(start_time.hour) + "hour.csv"

            print("fname=", root_path+fname)
            if os.path.isfile(root_path+fname):
                print("Reading file =>", fname)
                df = pd.read_csv(root_path+fname, sep="|")
                for j in range(0, len(df)):
                    segmentation = [lang + "_" +seg_name for seg_name in df["text"].iloc[j].split(" ")]
                    for seg in segmentation:
                        if not seg in segments:
                            segments[seg] = Segment(seg)
                        tweet_id = df["id"].iloc[j]
                        date = df["date"].iloc[j].split("_")[0]
                        hour = df["date"].iloc[j].split("_")[1].split(":")[0]
                        minutes = df["date"].iloc[j].split("_")[1].split(":")[1]
                        text = df["text"].iloc[j]
                        username = df["username"].iloc[j]
                        segments[seg].add_tweet(Tweet(tweet_id, date, hour, minutes, text, username))
                print("Number of Tweets Found: ", len(df))
            #else:
            #    print("Could not find file ", root_path+fname)

            # Increase the subwindow by 1 hour
            start_time += datetime.timedelta(hours=1)
        return segments

    def get_bursty_segments(self, time_window, seg_prob):
        """

        :param time_window: TimeWindow of our interest
        :param seg_prob: dictionary of segment probabilities
        :return:
        """
        segments_burst_w = []
        for seg_name in time_window.get_segment_names():
            tweets_count = 0
            user_set = set()
            for sw in time_window.subwindows:
                segment = sw.segments.get(seg_name, None)
                if segment is not None:
                    tweets_count += segment.tweet_count
                    user_set = user_set.union(segment.user_set)
            user_count = len(user_set)

            prob = seg_prob.get(seg_name, -1)

            if prob == -1:
                print('NEW SEGMENT:', seg_name)
                bursty_score = log10(1+user_count)
            else:
                seg_mean = tweets_count * prob
                seg_std_dev = sqrt(tweets_count * prob * (1 - prob))
                bursty_score = sigmoid(10 * (tweets_count - seg_mean - seg_std_dev)/(seg_std_dev)) * log10(1+user_count)

            #Segment(seg_name)
            segments_burst_w.append((seg_name, bursty_score))

        return segments_burst_w


if __name__ == "__main__":

    args = get_args()

    bs = BurstySegmentation()
    root_dir = "/aimlx/Datasets/EvDet/world_cup_18/csv_files/"
    languages = ["de", "en", "es", "fr", "it", "pt", "pl", "ru"]
    time_window_size = args.time_window_size   # 2 hours

    print(time_window_size)
    sub_window_size = args.sub_window_size   # 1 hours
    start = datetime.datetime(2018, 6, 14, 15, 0, 0)
    start_time = start
    end_time = datetime.datetime(2018, 6, 15, 0, 0, 0)
    n_time_windows = args.n_time_windows  # 4

    print("n_time_windows:", n_time_windows)

    """ 1. Splitting tweets into subwindows """
    for lang in languages:
        cleaned_tweet_dir = root_dir + "cleaned_tweets/" +lang + "/"
        sub_windows_dir = cleaned_tweet_dir + "sub-windows/"
        print(" Splitting Tweets per hour for " + lang + " ...")
        bs.split_tweets_time_windows(cleaned_tweet_dir, sub_windows_dir)

    """ 2. Iterating over the subwindows for all languages at once and creating TimeWindows 
    to compute probability of each segment in any random time interval """
    for lang in languages:
        seg_prob_lang = {}# dictionary per language
        print("Processing language: ", lang)
        cleaned_tweet_dir = root_dir + "cleaned_tweets/" +lang + "/"
        sub_windows_dir = cleaned_tweet_dir + "sub-windows/"

        seg_prob_tw = []  # list per timeWindow of dictionaries of probabilities per seg_name

        # Initialize the TimeWindow
        subwindows = []
        n_subwindows = int(time_window_size/sub_window_size)
        print("n_subwindows=", n_subwindows)
        for sub_window_no in range(n_subwindows):
            sw = bs.read_subwindow(sub_windows_dir, start_time, sub_window_size, lang)
            start_time += datetime.timedelta(hours=sub_window_size)
            subwindows.append(sw)

        tw = TimeWindow(subwindows)

        print("Computing Segment probabilities in Time Windows separately")
        for i in tqdm(range(n_time_windows)):
            print("n_subwindows:", n_subwindows)
            for j in range(n_subwindows):
                print("Iteration:", j)
                print("start_time:", start_time)
                print("Compute Segment probabilities in that time window for len(segments): ", len(tw.get_segment_names()))
                seg_prob = {}
                #print("tw=", tw.to_str())
                for seg_name in tqdm(tw.get_segment_names()):
                    seg_prob[seg_name] = tw.get_segment_prob_time_window(seg_name)
                seg_prob_tw.append(seg_prob)

                if start_time == end_time:
                    print("Reached the end ")
                    break

                # Advance the Time Window
                tw.advance_window(bs.read_subwindow(sub_windows_dir, start_time, sub_window_size, lang))
                start_time += datetime.timedelta(hours=sub_window_size)

        print("Taking the average over all TimeWindows")
        # Average of probabilities over all TimeWindows
        seg_prob_list = {}
        for i in range(n_time_windows):
            for seg in seg_prob_tw[i]:
                if seg not in seg_prob_list:
                    seg_prob_list[seg] = []
                seg_prob_list[seg].append(seg_prob_tw[i][seg])

        for seg in seg_prob_list:
            seg_prob_avg = np.mean(seg_prob_list[seg])
            seg_prob_lang[seg] = seg_prob_avg

        print("Saving seg_prob_lang ")
        with open(cleaned_tweet_dir+"seg_prob.p", "wb") as file:
            pkl.dump(seg_prob_lang, file)

        start_time = start
