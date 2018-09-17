import json

lang = "en"
dir_ = "/aimlx/Datasets/EvDet/world_cup_18/original_tweets/teamTags1/15_6_2018/"
file1 = "fetched_tweets_" + lang + "15-6_9hour.txt"
file2 = "fetched_tweets_" + lang + "_15-6_9hour.txt"

with open(dir_+file1) as f1:
    l1 = json.load(f1)

dict_l = {}
for el in l1:
    hour = el["created_at"].split(" ")[3].split(":")[0]
    if hour not in dict_l:
        dict_l.update({hour: [el]})
    else:
        list_1 = dict_l[hour]
        list_1.append(el)
        dict_l.update({hour: list_1})

"""
with open(dir_+file2) as f2:
    l2 = json.load(f2)

for el in l2:
    hour = el["created_at"].split(" ")[3].split(":")[0]
    if hour not in dict_l:
        dict_l.update({hour:[el]})
    else:
        list_1 = dict_l[hour]
        list_1.append(el)
        dict_l.update({hour:list_1})
"""

l_t = []
for hour in dict_l:
    print(str(hour))
    with open(dir_+"fetched_tweets_" + lang + "_14-6_"+str(hour)+"hour.txt", mode='w') as f:
        f.write(json.dumps(dict_l[hour], indent=2))
    """
    for el in dict_l[hour]:
        hour1 = el["created_at"].split(" ")[3].split(":")[0]
        if hour1 != hour:
            print("Something wrong")
    """


