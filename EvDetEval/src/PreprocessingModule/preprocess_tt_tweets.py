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

import time
import os
import pandas as pd
from tqdm import tqdm
import re

from nltk.corpus import stopwords
from treetagger import TreeTagger
import string

global tree_tagger_langs
tree_tagger_langs = ['pl', 'ru']
emoji_pattern = re.compile(
    u"(\ud83d[\ude00-\ude4f])|"  # emoticons
    u"(\ud83c[\udf00-\uffff])|"  # symbols & pictographs (1 of 2)
    u"(\ud83d[\u0000-\uddff])|"  # symbols & pictographs (2 of 2)
    u"(\ud83d[\ude80-\udeff])|"  # transport & map symbols
    u"(\ud83c[\udde0-\uddff])"  # flags (iOS)
    "+", flags=re.UNICODE)

from prepare_csv_data import *
stop_dict = {"ru": "russian", "pl": "polish"}

def compound_word_split(compound_word):
    """
    Split a given compound word and return list of words in given compound_word
    Ex: 'pyTWEETCleaner' --> ['py', 'TWEET', 'Cleaner']
    """
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', compound_word)
    return [re.sub(r'\d+', '', m.group(0).lower()) for m in matches if len(re.sub(r'\d+', '', m.group(0).lower()))>2]


class TweetPreprocessor():
    def __init__(self, date, lang, lemmatize_words=True, remove_stopwords=True, keep_only_nava= True, remove_retweets=False):
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
        if not os.path.isdir(target_dir):
            os.mkdir(target_dir)
        for dir_path, sub_dir_list, file_list in os.walk(root_dir):
            for fname in file_list:
                if fname.split(".")[-1] == "csv" and fname.split("_")[0] in tree_tagger_langs and fname.split("_")[0] \
                        == self.lang and fname.split(".")[0][3:] == self.date:
                    tree_tagger_models = {"pl": "polish", "ru": "russian"}
                    print("Preprocessing ", fname)
                    lang = fname.split("_")[0]
                    print("tree_tagger_models[lang]=", tree_tagger_models[lang])
                    nlp = TreeTagger(language=tree_tagger_models[lang])
                    df = pd.read_csv(dir_path + fname, sep="|", error_bad_lines=False)
                    texts = []
                    cleaned_texts = []
                    usernames = []
                    created_at = []
                    ids = []
                    df = df[pd.notnull(df['text'])]
                    for i in tqdm(range(0, len(df))):
                        text = df["text"].iloc[i]
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
                            if text.startswith("RT @") or text.startswith("@"):
                                cleaned_text = self.clean_text(text[text.index(' ')+2:], lang, nlp)
                            else:
                                cleaned_text = self.clean_text(text, lang, nlp)
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
        doc = nlp.tag(text)
        for token in doc:
            if not regexp.search(token[0]) and not token[2] not in set(stopwords.words(stop_dict[lang])) and\
                            len(re.sub(r'\d+', '', token[2])) > 2 and token[2] not in string.punctuation:
                if token[2].startswith("splitstart"):
                    tokens += compound_word_split(token[2][10:])
                else:
                    tokens.append(re.sub(r'\d+', '', token[2].lower()))

        return " ".join(tokens)


if __name__ == "__main__":
    st_time = time.time()

    root_dir = "/Users/meryemmhamdi/Documents/meryemRig1/home/meryem/meryem/Datasets" \
               "/EvDet/world_cup_18/csv_files/"
    original_tweets_dir = root_dir + "original_tweets/"

    cleaned_tweet_dir = "cleaned_tweets/" #root_dir + "cleaned_tweets/"

    orig_dir = "/Users/meryemmhamdi/Documents/meryemRig1/home/meryem/meryem/Datasets/EvDet/world_cup_18/"

    for lang in ["ru", "pl"]:
        for date in ["14_6_2018", "15_6_2018", "16_6_2018", "17_6_2018", "18_6_2018", "19_6_2018", "20_6_2018",
                     "20_6_2018", "21_6_2018", "22_6_2018", "23_6_2018", "24_6_2018", "25_6_2018", "26_6_2018",
                     "27_6_2018", "28_6_2018", "29_6_2018", "30_6_2018"]:
            print("Preparing dataset for lang=> " + lang + " date=> ", date)
            day = int(date.split("_")[0])
            month = int(date.split("_")[1])
            year = int(date.split("_")[2])
            dp = DataPreparator(day, month, year, 0, [lang], orig_dir, root_dir, ["allTags"])
            dp.json_to_csv()

            tp = TweetPreprocessor(date, lang)
            tp.clean_tweets(original_tweets_dir, cleaned_tweet_dir)


    """
    nlp = TreeTagger(language="russian")
    cleaned_text = tp.clean_text("@амин Я хочу посмотреть футбол сегодня", "ru", nlp)
    print(cleaned_text)
    """