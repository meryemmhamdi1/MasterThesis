from __future__ import absolute_import, print_function

from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import os
import argparse
from httplib import IncompleteRead
from urllib3.exceptions import ProtocolError, ReadTimeoutError
from requests.exceptions import ConnectionError
import sys
import datetime as dt
import yaml
import json


def read_config_yml(filename):
    stram = open(filename, "r")
    str_filters = yaml.load(stram)
    consumer_key = {}
    consumer_secret = {}
    access_token = {}
    access_token_secret = {}
    for key in str_filters["LANGS"]:
        langs = key.split("_")
        for lang in langs:
            consumer_key.update({lang.lower(): str_filters["LANGS"][key]["api_key"]})
            consumer_secret.update({lang.lower(): str_filters["LANGS"][key]["api_secret"]})
            access_token.update({lang.lower(): str_filters["LANGS"][key]["access_token"]})
            access_token_secret.update({lang.lower(): str_filters["LANGS"][key]["access_secret"]})
    return consumer_key, consumer_secret, access_token, access_token_secret

def get_args():
    """
    Command line arguments

    Arguments set the default values of command line arguments
    :return:
    """
    parser = argparse.ArgumentParser()

    """ Dataset Path Parameters """
    parser.add_argument("--save-path", "-dc", type=str, default="/aimlx/Datasets/EvDet/world_cup_18/original_tweets/",
                        help="where the crawled data is going to be saved")

    parser.add_argument("--crawler-mode", "-cm", type=str, default="allTags",
                        help="mode of crawling")

    parser.add_argument("--lang", "-l", type=str, default="it",
                        help="Other choices: ar, da,"
                             " de, en,"
                             " es, fa,"
                             " fr, hr,"
                             " is, it,"
                             " ja, ko,"
                             " pl, pt,"
                             " ru, sr,"
                             "sv")

    return parser.parse_args()


class StdOutListener(StreamListener):
    """ A listener handles tweets that are received from the stream.
    This is a basic listener that just prints received tweets to stdout.

    """
    def __init__(self, args):
        self.count = 0
        self.args = args

    def on_data(self, data):
        keep = False
        data_ = json.loads(data)
        if 'in_reply_to_status_id' in data_.keys():
            in_reply_to = data_['in_reply_to_status_id']
            if in_reply_to is not None:
                keep = True
        else:
            if 'id' in data_.keys() and 'user' in data_.keys() and 'text' in data_.keys() \
                    and 'lang' in data_.keys() and 'created_at' in data_.keys():
                keep = True

        if keep:
            date = str(dt.datetime.now().day) + "_" + str(dt.datetime.now().month) + "_" + str(dt.datetime.now().year)
            dirname = self.args.save_path + "allTags/" + date + "/" #self.args.crawler_mode

            if not os.path.isdir(dirname):
                os.mkdir(dirname)

            timedate = str(dt.datetime.now().day) + "-" + str(dt.datetime.now().month) + "_" + str(dt.datetime.now().hour) + "hour"
            fname = dirname + 'fetched_tweets_'+data_["lang"] + "_" + timedate+'.txt'

            with open(fname, "a") as file:
                file.write(data)

            self.count += 1
            if self.count % 10 == 0:
                print(self.count)

        return True

    def on_error(self, status):
        print("error:", status)


if __name__ == '__main__':
    global stored_exception
    stored_exception = None
    args = get_args()

    if not os.path.isdir(args.save_path):
        os.mkdir(args.save_path)

    if not os.path.isdir(args.save_path+args.crawler_mode):
        os.mkdir(args.save_path+args.crawler_mode)

    l = StdOutListener(args)

    consumer_key, consumer_secret, access_token, access_token_secret = read_config_yml("stream_config.yml")
    auth = OAuthHandler(consumer_key[args.lang], consumer_secret[args.lang])
    auth.set_access_token(access_token[args.lang], access_token_secret[args.lang])

    hashtags = []
    with open("worldCup18/officialTags", "r") as file:
        for line in file:
            hashtags.append(line.strip("\n").decode("utf-8"))

    with open("worldCup18/teamTags1", "r") as file:
        for line in file:
            hashtags.append(line.strip("\n").decode("utf-8"))

    with open("worldCup18/teamTags2", "r") as file:
        for line in file:
            hashtags.append(line.strip("\n").decode("utf-8"))

    while True:
        try:
            stream = Stream(auth, l)
            stream.filter(track=hashtags, languages=[args.lang])  #lang_list, follow=['138372303'] # officialTags #Sam Green  #officialTags, '138372303'
        except (IncompleteRead, ProtocolError, ReadTimeoutError, ConnectionError):  # Exception as e:
            print("HTTP/Request Error. Continuing ....")
            continue
        if stored_exception:
            raise stored_exception[0], stored_exception[1], stored_exception[2]

        sys.exit()
