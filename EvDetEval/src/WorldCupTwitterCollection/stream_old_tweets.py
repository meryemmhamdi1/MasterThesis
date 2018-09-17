import tweepy
import argparse
import time
from authentication import *

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

    parser.add_argument("--crawler-mode", "-cm", type=str, default="officialTags",
                        help="Other choices: officialTags, teamTags1, teamTags2")

    return parser.parse_args()



OAUTH_KEYS = {'consumer_key': consumer_key, 'consumer_secret': consumer_secret,
              'access_token_key': access_token, 'access_token_secret': access_token_secret}
auth = tweepy.OAuthHandler(OAUTH_KEYS['consumer_key'], OAUTH_KEYS['consumer_secret'])
api = tweepy.API(auth)

# Extract the first "xxx" tweets related to "fast car"
hashtags = []
args = get_args()
with open("worldCup18/"+args.crawler_mode, "r") as file:
    for line in file:
        hashtags.append(line.strip("\n").decode("utf-8"))

#print(hashtags)
hashtags_str = " OR ".join(hashtags).encode("utf-8")
print(hashtags_str)

while True:
    for tweet in tweepy.Cursor(api.search, q="worlcup", since='2018-06-13', until='2018-06-14', languages=["fr"]).items():#.items(200): # need to figure out how to extract all tweets in the previous day
        #if tweet.geo != None:
        print "////////////////////////////////"
        print "Tweet text:", tweet.text.encode("utf-8")
        print "Tweet lang:", tweet.lang


