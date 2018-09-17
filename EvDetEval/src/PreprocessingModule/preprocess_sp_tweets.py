# encoding=utf8
#import sys
#reload(sys)
#sys.setdefaultencoding("utf8")
"""
This file is the main file for testing and executing preprocessing of tweets in all languages
where each file is in one language and saves the output in a csv file for each language.

Details of cleaning:
        Language Identification
        Tokenization and Lemmatization
        Stopword removal
        Keeping only NAVA words
        Remove retweet mentions like RT and @username mentions
        Remove emojis
        Remove hashtags
        Dependency Parsing to treat negation, modifiers

"""
from __future__ import absolute_import
import time
import spacy
import os
import pandas as pd
from tqdm import tqdm
import re
import math

try:
    from spacy.lang.en.stop_words import STOP_WORDS as en_stop_words
    from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop_words
    from spacy.lang.it.stop_words import STOP_WORDS as it_stop_words
    from spacy.lang.de.stop_words import STOP_WORDS as de_stop_words
    from spacy.lang.pt.stop_words import STOP_WORDS as pt_stop_words
    from spacy.lang.es.stop_words import STOP_WORDS as es_stop_words
except ImportError:
    en_stop_words = fr_stop_words = it_stop_words = de_stop_words = pt_stop_words = es_stop_words = ["the"]

from nltk.corpus import stopwords
import numpy as np

from prepare_csv_data import *

global spacy_langs
spacy_langs = ['fr', 'es', 'de', 'it',  'en', 'pt']
emoji_pattern = re.compile(
    u"(\ud83d[\ude00-\ude4f])|"  # emoticons
    u"(\ud83c[\udf00-\uffff])|"  # symbols & pictographs (1 of 2)
    u"(\ud83d[\u0000-\uddff])|"  # symbols & pictographs (2 of 2)
    u"(\ud83d[\ude80-\udeff])|"  # transport & map symbols
    u"(\ud83c[\udde0-\uddff])"  # flags (iOS)
    "+", flags=re.UNICODE)

global stopwords
stopwords = {"en": en_stop_words, "fr": fr_stop_words, "de": de_stop_words, "it": it_stop_words,
             "es": es_stop_words, "pt": pt_stop_words}

global stopwords_all
stopwords_all = list(en_stop_words) + list(fr_stop_words) + list(de_stop_words) + list(it_stop_words) +\
                list(es_stop_words) + list(pt_stop_words)


def compound_word_split(compound_word):
    """
    Split a given compound word and return list of words in given compound_word
    Ex: 'pyTWEETCleaner' --> ['py', 'TWEET', 'Cleaner']
    """
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', compound_word)
    return [re.sub(r'\d+', '', m.group(0).lower()) for m in matches if len(re.sub(r'\d+', '', m.group(0).lower()))>2]


class TweetPreprocessor():
    def __init__(self, date, lang, lemmatize_words=True, remove_stopwords=True, keep_only_nava= True, remove_retweets=False):
        print("TweetPreprocessor")
        self.date = date
        self.lang = lang
        self.lemmatize_words = lemmatize_words
        self.remove_stopwords = remove_stopwords
        self.keep_only_nava = keep_only_nava
        self.remove_retweets = remove_retweets

    def clean_tweets(self, root_dir, target_dir):
        """
        Iterates over the files in original_tweets folder to preprocess each file
        :return:
        """
        # Create target directory if it doesn't exist
        z = lambda line: re.compile('\#').sub('', re.compile('.*?@\s*\w*:*\s*').sub('', line, count=1).strip())
        if not os.path.isdir(target_dir):
            os.mkdir(target_dir)
        for dir_path, sub_dir_list, file_list in os.walk(root_dir):
            for fname in file_list:
                if fname.split(".")[-1] == "csv" and fname.split("_")[0] in spacy_langs and fname.split("_")[0] \
                        == self.lang and fname.split(".")[0][3:] == self.date:
                    print("PREPROCESSING AND TREATING: ", fname)
                    lang = fname.split("_")[0]
                    nlp = spacy.load(lang)
                    df = pd.read_csv(dir_path + fname, sep="|", error_bad_lines=False, encoding="utf-8")
                    texts = []
                    cleaned_texts = []
                    usernames = []
                    created_at = []
                    ids = []
                    df = df[pd.notnull(df['text'])]
                    for i in tqdm(range(0, len(df))):
                        text = df["text"].iloc[i]
                        # print("text:", text)
                        text = text.replace('#', 'splitstart').strip("\n") # to split officialTags after removing punctuations
                        if self.remove_retweets:
                            if not text.startswith("RT @") and not text.startswith("@"):
                                cleaned_text = self.clean_text(text, lang, nlp)
                                cleaned_texts.append(cleaned_text)
                                texts.append(text)
                                usernames.append(df["user_name"].iloc[i])
                                created_at.append(df["date"].iloc[i])
                                ids.append(df["id"].iloc[i])
                        else:
                            if text.startswith("RT @"):
                                text = text.replace("RT ", "")
                                cleaned_text = self.clean_text(z(text), lang, nlp)
                            elif "@" in text:
                                cleaned_text = self.clean_text(z(text), lang, nlp)
                            else:
                                cleaned_text = self.clean_text(text, lang, nlp)
                                """
                                if " " in text[4:]:
                                    cleaned_text = self.clean_text(text[text.index(' ', 3)+1:], lang, nlp)
                                else:
                                    cleaned_text = self.clean_text(text[4:])
                                elif text.startswith("@"):
                                    if " " in text[1:]:
                                        cleaned_text = self.clean_text(text[text.index(' ')+1:], lang, nlp)
                                    else:
                                        cleaned_text = self.clean_text(text[2:])
                                """
                            texts.append(text)
                            cleaned_texts.append(cleaned_text)
                            usernames.append(df["user_name"].iloc[i])
                            created_at.append(df["date"].iloc[i])
                            ids.append(df["id"].iloc[i])

                    # Save the newly preprocessed file for each language independently
                    cleaned_df = pd.DataFrame()
                    cleaned_df["id"] = ids
                    cleaned_df["username"] = usernames
                    cleaned_df["date"] = created_at
                    cleaned_df["text"] = cleaned_texts
                    cleaned_df["original_text"] = texts

                    print("Saving cleaned tweets ... ")

                    save_path = target_dir+lang+"/"
                    if not os.path.isdir(save_path):
                        os.mkdir(save_path)

                    cleaned_df.to_csv(save_path+fname, sep="|", encoding="utf-8")

    def clean_text(self, text, lang, nlp):
        """
        Run Spacy preprocessing on text
        :param text:
        :return:
        """

        text = emoji_pattern.sub(r'', text)
        tokens = []
        text = text.replace("http:// ", "http://").replace("https:// ", "https://")
        url_reg = r'^(http:\/\/www\.|https:\/\/www\.|http:\/\/|https:\/\/)?[a-z0-9]+([\-\.]{1}[a-z0-9]+)*\.[a-z]{2,5}(:[0-9]{1,5})?(\/.*)?'
        regexp = re.compile(url_reg)
        doc = nlp(unicode(text))
        for token in doc:
            if not token.like_url and not regexp.search(token.lemma_) and not token.is_punct and token.lemma_ not in stopwords_all \
                    and len(re.sub(r'\d+', '', token.lemma_)) > 2:  # remove hyperlinks, punctuation and stopwords
                if token.lemma_.startswith("splitstart"):
                    tokens += compound_word_split(token.lemma_[10:])
                else:
                    if token.lemma_ == "-PRON-":
                        tokens.append(re.sub(r'\d+', '', token.norm_.lower()))
                    else:
                        tokens.append(re.sub(r'\d+', '', token.lemma_.lower()))
        return " ".join(tokens)


if __name__ == "__main__":
    st_time = time.time()

    root_dir = "/aimlx/Datasets/EvDet/world_cup_18/csv_files/"#"/aimlx/Datasets/EvDet/world_cup_14/"
    original_tweets_dir = root_dir + "original_tweets/"

    cleaned_tweet_dir = root_dir + "cleaned_tweets/"
    orig_dir = "/aimlx/Datasets/EvDet/world_cup_18/"

    for lang in ["es"]:# "it", "de", "pt"
        for date in ["21_6_2018","22_6_2018", "23_6_2018", "24_6_2018", "25_6_2018", "26_6_2018", "27_6_2018", "28_6_2016", "29_6_2018"]:
            print("Preparing dataset for lang=> " + lang + " date=> ", date)
            day = int(date.split("_")[0])
            month = int(date.split("_")[1])
            year = int(date.split("_")[2])
            dp = DataPreparator(day, month, year, 0, [lang], orig_dir, root_dir, ["allTags"])
            dp.json_to_csv()

            tp = TweetPreprocessor(date, lang)
            tp.clean_tweets(original_tweets_dir, cleaned_tweet_dir)

    """
    nlp = spacy.load("en")

    cleaned_text = tp.clean_text("@ActuFoot_ : #Humour : Messi recevant le Ballon d'Or de la Coupe du Monde 2014",
                                 "fr", nlp)
    print(cleaned_text)
    """
