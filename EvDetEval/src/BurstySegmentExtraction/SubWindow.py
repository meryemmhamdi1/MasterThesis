class SubWindow():
    """
    This class for each subwindow and the Segment it contains.
    Each SubWindow = t time interval
    """
    time_frame_counter = 0

    def __init__(self, segments):
        SubWindow.time_frame_counter += 1
        self.segments = segments
        self.time_frame_counter = SubWindow.time_frame_counter

    def __str__(self):
        result = 'SubWindow #'+str(self.time_frame_counter)+', No. of Tweets: '+str(self.get_total_tweet_count())
        return result

    def get_segments(self):
        return self.segments

    def get_segment_names(self):
        segment_names = set()
        for seg in self.segments:
            segment_names.add(seg)
        return segment_names

    def append_subwindow(self, segments_2):
        segments = self.segments.union(segments_2)
        return SubWindow(segments)

    def get_total_tweet_count(self):
        total_tweet_count = 0
        for seg in self.segments:
            total_tweet_count += self.segments[seg].tweet_count

        return total_tweet_count

    def get_tweets_containing_segments(self, seg_name):
        if seg_name not in self.segments:
            print(seg_name)
        segment = self.segments.get(seg_name, None)
        if segment is not None:
            return segment.tweets_list
        else:
            return None

    def get_freq_of_segment(self, segment):
        seg_freq = 0
        if segment in self.segments:
            seg_freq += self.segments[segment].tweet_count
        return seg_freq

    def get_user_count_for_segment(self, segment):
        return self.segments[segment].get_user_count()



