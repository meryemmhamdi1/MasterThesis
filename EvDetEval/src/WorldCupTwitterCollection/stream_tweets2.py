from __future__ import absolute_import, print_function

from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
import tweepy
from tweepy import Stream
from authentication2 import *
import time
import json
import os
import argparse
from httplib import IncompleteRead
from urllib3.exceptions import ProtocolError, ReadTimeoutError
from requests.exceptions import ConnectionError
from exceptions import Exception
import sys
import datetime as dt

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

    parser.add_argument("--crawler-mode", "-cm", type=str, default="teamTags2",
                        help="Other choices: officialTags, teamTags1, teamTags2")

    return parser.parse_args()


class StdOutListener(StreamListener):
    """ A listener handles tweets that are received from the stream.
    This is a basic listener that just prints received tweets to stdout.

    """
    def __init__(self, args):
        self.data_lang = {}
        self.args = args

    def on_data(self, data):
        """
        with open('/aimlx/Datasets/EvDet/world_cup_18/fetched_tweets.txt', 'a') as tf:
            tf.write(data)
        """
        #print("User id: ", json.loads(data)["user"]["id"])
        #print("Username:", json.loads(data)["user"]["name"])
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
            if data_['lang'] not in self.data_lang:
                self.data_lang[data_['lang']] = 1
            else:
                self.data_lang[data_['lang']] += 1

            print(self.data_lang)

            a = []
            date = str(dt.datetime.now().day) + "_" + str(dt.datetime.now().month) + "_" + str(dt.datetime.now().year)
            dirname = self.args.save_path + self.args.crawler_mode + "/" + date + "/"

            if not os.path.isdir(dirname):
                os.mkdir(dirname)

            timedate = str(dt.datetime.now().day) + "-" + str(dt.datetime.now().month) + "_" + str(dt.datetime.now().hour) + "hour"
            fname = dirname + 'fetched_tweets_'+data_["lang"] + "_" + timedate+'.txt'

            print("Started Storing .................")
            if not os.path.isfile(fname):
                a.append(data_)
                with open(fname, mode='w') as f:
                    f.write(json.dumps(a, indent=2))
            else:
                with open(fname) as feedsjson:
                    feeds = json.load(feedsjson)

                feeds.append(data_)
                with open(fname, mode='w') as f:
                    f.write(json.dumps(feeds, indent=2))
            print("Finished Storing ....")

            """
            with open(fname, "a") as file:
                if data_["in_reply_to_status_id"] is None:
                    in_reply_to_status_id = "None"
                else:
                    in_reply_to_status_id = str(data_["in_reply_to_status_id"])
                if data_["retweeted"]:
                    retweeted = "True"
                else:
                    retweeted = "False"

                if data_["geo"] is None:
                    geo = "None"
                else:
                    geo = str(data_["geo"])

                if data_["place"] is None:
                    place = "None"
                else:
                    place = str(data_["place"])

                if data_["coordinates"] is None:
                    coordinates = "None"
                else:
                    coordinates = str(data_["coordinates"])

                file.write(data_["text"].encode("utf-8") + "||" + in_reply_to_status_id+"||" + retweeted+"||" +data_["source"]+"||" +str(data_["retweet_count"])+ "||" +data_["id_str"]+ "||" + data_["user"]["name"]+ "||" +str(data_["user"]["id"])+ "||" +data_["lang"]+ "||" +data_["created_at"]+ "||" +data_["timestamp_ms"]+ "||" +geo+ "||" +place+ "||" +coordinates+ "||" )
            """

        return True

    def on_error(self, status):
        print("error:", status)

if __name__ == '__main__':
    global stored_exception
    stored_exception= None
    args = get_args()

    if not os.path.isdir(args.save_path):
        os.mkdir(args.save_path)

    if not os.path.isdir(args.save_path+args.crawler_mode):
        os.mkdir(args.save_path+args.crawler_mode)

    l = StdOutListener(args)
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    lang_list = []
    with open("worldCup18/languages", "r") as file:
        for line in file:
            lang_list.append(line.strip("\n"))

    hashtags = []
    with open("worldCup18/"+args.crawler_mode, "r") as file:
        for line in file:
            hashtags.append(line.strip("\n").decode("utf-8"))

    while True:
        try:
            stream = Stream(auth, l)
            stream.filter(track=hashtags, languages=lang_list) #lang_list, follow=['138372303'] # officialTags #Sam Green  #officialTags, '138372303'
        except (IncompleteRead, ProtocolError, ReadTimeoutError, ConnectionError):# Exception as e:
            # Oh well, reconnect and keep trucking
            print("HTTP/Request Error. Continuing ....")
            continue
        """
        except KeyboardInterrupt:
            print("Keyboard Interrupt")
            time.sleep(20)
            print("Keyboard Interrupt for REAL")
            stored_exception = sys.exc_info()
            # Or however you want to exit this loop
            #stream.disconnect()
            break
        """
        if stored_exception:
            raise stored_exception[0], stored_exception[1], stored_exception[2]

        sys.exit()
    """
    api = tweepy.API(auth)
    kwargs = {'screen_name': '@FIFAWorldCup', 'include_rts': 0, 'count': 1}
    #tweets = api.user_timeline(**kwargs) #tweet object has tweets of user
    for status in tweepy.Cursor(api.user_timeline, track=["WorldCup"], screen_name='@FIFAWorldCup', include_rts=0, count=1).items():
        print("username:", status._json['user']['name'].encode("utf-8"))
        print("userid:", status._json['user']['id'])
        print("text:", status._json['text'].encode("utf-8"))
        print("------------------------------------------")

    username = json.loads(tweets)["user"]["name"]
    text = json.loads(tweets[0])["text"]
    userid = json.loads(tweets[0])["user"]["id"]
    print("username:", username.encode("utf-8"))
    print("user_id:", userid)
    print("text:", text.encode("utf-8"))
    
    """



"""
The tweepy does not provide the "since" argument, as you can check yourself here.

To achieve the desired output, you will have to use the api.user_timeline, iterating through pages until the desired date is reached, Eg:

import tweepy
import datetime

# The consumer keys can be found on your application's Details
# page located at https://dev.twitter.com/apps (under "OAuth settings")
consumer_key=""
consumer_secret=""

# The access tokens can be found on your applications's Details
# page located at https://dev.twitter.com/apps (located
# under "Your access token")
access_token=""
access_token_secret=""

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)
page = 1
stop_loop = False
while not stop_loop:
    tweets = api.user_timeline(username, page=page)
    if not tweets:
        break
    for tweet in tweets:
        if datetime.date(YEAR, MONTH, DAY) < tweet.created_at:
            stop_loop = True
            break
        # Do the tweet process here
    page+=1
    time.sleep(500)
"""
