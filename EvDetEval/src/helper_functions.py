import types

from BurstySegmentExtraction.detect_bursty_segments import *
#from PreprocessingModule.prepare_csv_data import *
#from PreprocessingModule.preprocess_sp_tweets import *
from EventSegmentClustering.cluster_events import *
import datetime as dt

n_time_windows = 1 
time_window_size = 2
sub_window_size = 1  

bs = BurstySegmentation()
ed = EventDetector()

root_dir = "/Users/meryemmhamdi/Documents/Rig4/meryemRig4/home/meryem/meryem/Datasets/EvDet/world_cup_18/csv_files/"
def find_total_tweets(triggers, sws_time):
    counts = set()
    for j in range(len(triggers)):
        for i in range(len(sws_time)):
            for term in triggers[j]:
                if sws_time[i].get_tweets_containing_segments(term) is not None:
                    for tweet in sws_time[i].get_tweets_containing_segments(term):
                        counts.add(tweet.tweet_id)
    return len(counts)


def get_sws_times(lang_set, start):
    sws = []
    start_time = start
    for lang in lang_set:
        print(">>>>>>>>>>>>>>> Preprocessing LANGUAGE:", lang)
        cleaned_tweet_dir = root_dir + "cleaned_tweets/" +lang + "/"
        sub_windows_dir = cleaned_tweet_dir + "sub-windows/"

        # Initialize the TimeWindow
        subwindows = []
        segments_subs = []
        n_subwindows = int(time_window_size/sub_window_size)
        for sub_window_no in range(n_subwindows):
            sw = bs.read_subwindow(sub_windows_dir, start_time, sub_window_size, lang)
            start_time += datetime.timedelta(hours=sub_window_size)
            subwindows.append(sw)
            segments_subs.append(sw.get_segments())

        tw = TimeWindow(subwindows)

        sws.append(segments_subs)
        start_time = start

    print("Aggregating the Time Windows=>")
    sws_langs = map(list, zip(*sws))
    sws_time = []

    for sws_lang in sws_langs:
        multi_lang_segment = union(*sws_lang)
        sws_time.append(SubWindow(multi_lang_segment))
    
    return sws_time

def find_counts(triggers, sws_time):
    counts = {}
    for j in range(len(triggers)):
        for i in range(len(sws_time)):
            for term in triggers[j]:
                if sws_time[i].get_tweets_containing_segments(term) is not None:
                    for tweet in sws_time[i].get_tweets_containing_segments(term):
                        date_time = tweet.date+ "_"+ tweet.hour+ ":"+ tweet.minutes  
                        year = int(tweet.date.split("-")[0])
                        month = int(tweet.date.split("-")[1])
                        day = int(tweet.date.split("-")[2])
                        hour = int(tweet.hour)
                        minutes = int(tweet.minutes)
                        dt_time = dt.datetime(year, month, day, hour, minutes, 0)
                        if j not in counts:
                            counts.update({j:{}})
                        if dt_time not in counts[j]:
                            counts[j].update({dt_time:1})
                        else:
                            counts[j].update({dt_time:counts[j][dt_time]+1})
                            
    new_counts = {}
    for i in range(len(counts)):
        max_value, max_index = np.max(list(counts[i].values())), np.argmax(list(counts[i].values()))
        min_value = np.min(list(counts[i].values()))
        new_counts.update({i:{}})
        for j in range(len(list(counts[i].values()))):
            key = list(counts[i].keys())[j]
            if j == max_index:
                new_counts[i].update({key:max_value})
            else:
                new_counts[i].update({key:min_value})
                
    return counts

def quant_metrics(counts, start): 
    count_events_1_std = set()
    count_events_2_std = set()
    count_events_3_std = set()
    
    list_counts = []
    for i in range(len(counts)):
        for el in list(counts[i].values()):
            list_counts.append(el)
            
    mean_y = np.mean(list_counts)
    std_y = np.std(list_counts)
    
    for i in range(len(counts)):
        mean_y = np.mean(list(counts[i].values()))
        std_y = np.std(list(counts[i].values()))
        x = list(counts[i].keys()) 
        y_c = list(counts[i].values())
        ## Ignoring Break time
        break_time_start = start + datetime.timedelta(minutes=48)
        break_time_end = break_time_start + datetime.timedelta(minutes=20)
        y_keep = []
        x_keep = []
        for j in range(len(y_c)):
            if x[j] < break_time_start or x[j] > break_time_end:
                x_keep.append(x[j])
                y_keep.append(y_c[j])
        flag = False
        for j in range(0,len(y_keep)):
            if y_keep[j] >= (mean_y + 3* std_y) and flag == False:
                count_events_3_std.add(x_keep[j])
            if y_keep[j] >= (mean_y + 2* std_y) and flag == False:
                count_events_2_std.add(x_keep[j])
            if y_keep[j] >= (mean_y + std_y) and flag== False:
                count_events_1_std.add(x_keep[j])
                flag = True
                
    return count_events_1_std, count_events_2_std, count_events_3_std

def event_within_range(true_event, det_event):
    margin = datetime.timedelta(minutes = 5)
    if true_event - margin <= det_event <=true_event + margin:
        return True
    return False

def get_correlated_events(std_datetimes):
    event_corr = {"goals":{}, "yellowCards":{}, "redCards": {}, "elimination": {}}
    events_corr_meta = {"goals":0, "yellowCards":0, "redCards": 0, "elimination": 0}
    flag_elim = False
    for event in std_datetimes:
        flag_done = False
        print("Event at:", event)
        for goal in goals_matches[match]:
            flag = event_within_range(goal, event)
            if flag:
                print("Goal Detected at:",goal)
                flag_done = True
                event_corr["goals"].update({goal:goals_matches[match][goal]})
                events_corr_meta.update({"goals":len(event_corr["goals"])})

        if not flag_done:
            for yellow_card in yellow_cards_matches[match]:
                flag = event_within_range(yellow_card, event)
                if flag:
                    print("Yellow Card Detected at:",yellow_card)
                    flag_done = True
                    event_corr["yellowCards"].update({yellow_card:yellow_cards_matches[match][yellow_card]})
                    events_corr_meta.update({"yellowCards":len(event_corr["yellowCards"])})

        if not flag_done:
            if match in red_cards_matches:
                for red_card in red_cards_matches[match]:
                    flag = event_within_range(red_card, event)
                    if flag:
                        print("Red Card Detected at:",red_card)
                        flag_done = True
                        event_corr["redCards"].update({red_card:red_cards_matches[match][red_card]})
                        events_corr_meta.update({"redCards":len(event_corr["redCards"])})
        
        if match in elimination and not flag_elim and not flag_done:
            for eli in elimination[match]:
                flag_elim = event_within_range(eli, event)
                if flag_elim:
                    print("Elimination Detected at:",eli)
                    flag_done = True
                    event_corr["elimination"].update({eli:elimination[match][eli]})
                    events_corr_meta.update({"elimination":events_corr_meta["elimination"]+1})
                    
    return event_corr, events_corr_meta

def flatten(T):
    if not isTuple(T): return (T,)
    elif len(T) == 0: return ()
    else: return flatten(T[0]) + flatten(T[1:]) 
    

def isTuple(x): 
    return type(x) == tuple


def find_counts_trigger(term, sws_time):
    counts = {}
    for i in range(len(sws_time)):
        if sws_time[i].get_tweets_containing_segments(term) is not None:
            for tweet in sws_time[i].get_tweets_containing_segments(term):
                date_time = tweet.date+ "_"+ tweet.hour+ ":"+ tweet.minutes  
                year = int(tweet.date.split("-")[0])
                month = int(tweet.date.split("-")[1])
                day = int(tweet.date.split("-")[2])
                hour = int(tweet.hour)
                minutes = int(tweet.minutes)
                dt_time = dt.datetime(year, month, day, hour, minutes, 0)
                if dt_time not in counts:
                    counts.update({dt_time:1})
                else:
                    counts.update({dt_time:counts[dt_time]+1})
                            
    new_counts = {}
    max_value, max_index = np.max(list(counts.values())), np.argmax(list(counts.values()))
    min_value = np.min(list(counts.values()))
    new_counts.update({i:{}})
    for j in range(len(list(counts.values()))):
        key = list(counts.keys())[j]
        if j == max_index:
            new_counts.update({key:max_value})
        else:
            new_counts.update({key:min_value})
                
    return counts, new_counts