import json

lang = "it"
dir_ = "/aimlx/Datasets/EvDet/world_cup_18/original_tweets/teamTags1/15_6_2018/"
file1 = "fetched_tweets_" + lang + "15-6_9hour.txt"
file2 = "fetched_tweets_" + lang + "_15-6_9hour.txt"
file_merged = "fetched_tweets_" + lang + "_15-6_9_NEW_hour.txt"

with open(dir_+file1) as f1:
    l1 = json.load(f1)

with open(dir_+file2) as f2:
    l2 = json.load(f2)

l_t = l1 + l2

with open(dir_+file_merged, mode='w') as f:
    f.write(json.dumps(l_t, indent=2))

