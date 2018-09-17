import json
with open("/aimlx/Datasets/EvDet/world_cup_18/original_tweets/allTags/19_6_2018/fetched_tweets_ko_19-6_15hour.txt") as file:
    for line in file:
        data = json.loads(line)
        #print("data.keys():", data.keys())
        print("username:", data["text"].encode("utf-8"))
