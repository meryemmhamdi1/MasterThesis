import datetime
from time import strptime
import os.path
import json
import pandas as pd


class DataPreparator():
    def __init__(self, day, month, year, hr, langs, data_path, save_path, hashtags):
        self.date = datetime.datetime(year, month, day, hr, 0, 0)
        self.hours = list(range(hr, 24, 1))
        self.langs = langs
        self.data_orig = data_path + "original_tweets/"
        self.data_save = save_path + "original_tweets/"
        self.dirs = []
        for hashtag in hashtags:
            self.dirs.append(self.data_orig+hashtag+"/")

    def json_to_csv(self):
        for lang in self.langs:
            print("LANGUAGE =====> ", lang)
            ids = []
            texts = []
            in_reply_to_status_ids = []
            retweets = []
            #userids = []
            usernames = []
            langs = []
            dates = []
            time_ints = []
            geos = []
            permalinks = []
            favorited = []
            for hour in self.hours:
                for dir in self.dirs:
                    fname = dir + str(self.date.day) + "_" + str(self.date.month) + "_" + str(self.date.year) \
                            + "/fetched_tweets_" + lang + "_" + str(self.date.day) + "-" + str(self.date.month) \
                            + "_" + str(hour) + "hour.txt"

                    print("fname=", fname)

                    if os.path.isfile(fname):
                        print("Processing fname= ", fname)
                        """
                        with open(fname) as file:
                            data = json.load(file)
                        """
                        data = []
                        with open(fname) as file:
                            for line in file:
                                data.append(json.loads(line))
                        for line in data:
                            ids.append(line["id"])
                            texts.append(line["text"])
                            if "in_reply_to_status_id" in line:
                                if line["in_reply_to_status_id"] is not None:
                                    in_reply_to_status_ids.append(str(line["in_reply_to_status_id"]))
                                else:
                                    in_reply_to_status_ids.append(None)
                            else:
                                if line["in_reply_to_status_ids"] is not None:
                                    in_reply_to_status_ids.append(str(line["in_reply_to_status_ids"]))
                                else:
                                    in_reply_to_status_ids.append(None)
                            retweets.append(str(line["retweet_count"]))
                            #userids.append(str(line["user"]["id"]))
                            usernames.append(line["user"]["name"])
                            langs.append(line["lang"])

                            day_j = line["created_at"].split(" ")[2]
                            mon_j = str(strptime(line["created_at"].split(" ")[1], '%b').tm_mon)
                            year_j = str(self.date.year)
                            time_j = line["created_at"].split(" ")[3]
                            time_int = int(line["created_at"].split(" ")[3].replace(":", ""))

                            dates.append(year_j+"-"+mon_j+"-"+day_j + " "+time_j)
                            time_ints.append(time_int)
                            if line["geo"] is not None:
                                geos.append(str(line["geo"]))
                            else:
                                geos.append(None)
                            permalinks.append(line["source"])
                            favorited.append(line["favorite_count"])

            # At the end, store everything in a dataframe
            if len(ids) > 0:  # Make sure tweets exists for that language within that timeframe
                df = pd.DataFrame()
                df["id"] = ids
                df["text"] = texts
                df["in_reply_to_status_ids"] = in_reply_to_status_ids
                df["retweet_count"] = retweets
                #df["user_id"] = userids
                df["user_name"] = usernames
                df["lang"] = langs
                df["date"] = dates
                df["time_int"] = time_ints
                df["geo"] = geos
                df["source"] = permalinks
                df["favorites"] = favorited

                result = df.sort_values(by=['time_int', 'id'])

                print("Saving in =>", self.data_save + lang+"_"+str(self.date.day) + "_" + str(self.date.month) + "_" +
                      str(self.date.year)+".csv")
                result.to_csv(self.data_save + lang+"_"+str(self.date.day) + "_" + str(self.date.month) + "_" +
                              str(self.date.year)+".csv", sep="|", encoding="utf-8")
