from __future__ import absolute_import
from __future__ import division
from .SubWindow import *
from .segment import *
from .tweet import *

class TimeWindow:
    """
    Time window during which events are continuously detected
    Each TimeWindow = m*t time interval
    """
    def __init__(self, ini_subwindows):
        """
        Initializing TimeWindow with the first subwindows
        """
        self.subwindows = ini_subwindows
        self.num_subwindows = len(ini_subwindows)
        self.start_frame = 1
        self.end_frame = self.num_subwindows

    def advance_window(self, next_subwindow):
        """
        Advance window with one subwindow
        :param next_subwindow:
        :return:
        """
        self.subwindows = self.subwindows[1:]
        self.subwindows.append(next_subwindow)
        self.start_frame += 1
        self.end_frame += 1

    def to_str(self):
        result = ''
        result += '----- TimeWindow['+str(self.start_frame)+'-'+str(self.end_frame)+'] -----\n'
        result += 'No. of Tweets: '+str(self.get_total_tweet_count())
        for sw in self.subwindows:
            result += '\n'+sw.__str__()
        result += '\n-----------------------------------------------'
        return result

    def get_segment_names(self):
        segment_names = set()
        for sw in self.subwindows:
            for seg in sw.get_segment_names():
                segment_names.add(seg)
        return segment_names

    def get_tweets_with_segment(self, segment):
        tweets = []
        for sw in self.subwindows:
            tweets_sw = sw.get_tweets_containing_segments(segment)
            if tweets_sw is not None:
                tweets += tweets_sw
        return tweets

    def get_total_tweet_count(self):
        total_tweets = 0
        for sw in self.subwindows:
            total_tweets += sw.get_total_tweet_count()
        return total_tweets

    def get_segment_prob_time_window(self, seg_name):
        seg_tweets_count = 0
        for sw in self.subwindows:
            seg_tweets_count += sw.get_freq_of_segment(seg_name)
        return seg_tweets_count /self.get_total_tweet_count()

    def get_segment_freq_time_window(self, seg_name):
        seg_tweets_count = 0
        for sw in self.subwindows:
            seg_tweets_count += sw.get_freq_of_segment(seg_name)
        return seg_tweets_count

""" Testing the functionalities of the code"""
if __name__ == "__main__":

    t1 = Tweet(488835330544766976, "2014-07-15", "01", "59", "good player world cup forget team need ahead mensajeamessi", "vivismelo27")
    t2 = Tweet(488835330393792512, "2014-07-15", "01", "59", "finish world cup 2014 come end", "_iandreea_")

    t3 = Tweet(488835330544766976, "2014-07-15", "01", "59", "good player world cup forget team need ahead mensajeamessi", "vivismelo27")
    t4 = Tweet(488835330393792512, "2014-07-15", "01", "59", "finish world cup 2014 come end", "meryem")

    t5 = Tweet(488835330544766976, "2014-07-15", "01", "59", "abc united airlines flight delayed", "_iandreea_")
    t6 = Tweet(488835330393792512, "2014-07-15", "01", "59", "def welcome united states flight passenger", "vivismelo27")
    t7 = Tweet(488835330544766976, "2014-07-15", "01", "59", "ghi today united airlines flight to united states cancelled", "meryem")

    t8 = Tweet(488835330393792512, "2014-07-15", "01", "59", "123 welcome united states flight passenger", "vivismelo27")
    t9 = Tweet(488835330544766976, "2014-07-15", "01", "59", "456 united airlines flight 10 am passenger welcome", "meryem")

    s1 = Segment('cup')
    for tweet in [t1, t2]:
        s1.add_tweet(tweet)

    s2 = Segment('world')
    for tweet in [t3, t4]:
        s2.add_tweet(tweet)

    s3 = Segment('flight')
    for tweet in [t5, t6, t7]:
        s3.add_tweet(tweet)

    s4 = Segment('welcome')
    for tweet in [t8, t9]:
        s4.add_tweet(tweet)

    s5 = Segment('passenger')
    for tweet in [t8, t9]:
        s5.add_tweet(tweet)

    segments_1 = {}
    segments_1['cup'] = s1
    segments_1['world'] = s2
    sw1 = SubWindow(segments=segments_1)

    segments_2 = {}
    segments_2['flight'] = s3
    sw2 = SubWindow(segments=segments_2)

    initial_subwindows = [sw1, sw2]
    t = TimeWindow(initial_subwindows)
    print(t)
    prob1 = t.get_segment_prob_time_window("cup")
    print("prob of cup ", prob1)

    segments_3 = {}
    segments_3['welcome'] = s4
    segments_3['passenger'] = s5
    sw3 = SubWindow(segments=segments_3)
    t.advance_window(sw3)
    print(t)

    prob2 = t.get_segment_prob_time_window("cup")
    print("prob of cup ", prob2)

    prob = (prob1 + prob2)/2
    print(">>>>>>>>>>>avg prob of cup ", prob)

