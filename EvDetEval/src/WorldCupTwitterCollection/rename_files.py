import os
import glob
dir_ = "/aimlx/Datasets/EvDet/world_cup_18/original_tweets/allTags/"
dates = ["10_6_2018", "11_6_2018", "12_6_2018", "13_6_2018", "14_6_2018", "15_6_2018", "16_6_2018", "17_6_2018", "18_6_2018", "19_6_2018"]

for date in dates:
    for file in glob.glob(dir_ + date + "/*.txt"):
        filename = file.split("/")[-1]
        lang = filename.split("_")[2]
        date_str = filename.split("_")[3]
        hour = filename.split("_")[4]
        print(lang + " hour= ", hour)
        filestr = dir_+date+"/fetched_tweets_"+lang+"_"+date_str+"_" +hour
        if hour[0] == "h":
            new_name = dir_+date+"/fetched_tweets_"+lang+"_"+date_str+"_" + "0"+ hour
            print("filename=", filestr, "renamed to:", new_name)
            os.rename(filestr, new_name)
