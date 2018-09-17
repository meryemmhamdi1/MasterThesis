class Segment():
    def __init__(self, segment):
        self.segment = segment
        self.tweets_list = []
        self.tweet_count = 0  # Number of tweets containing this segment
        self.user_set = set()
        self.bursty_score = 0

    def add_tweet(self, tweet):
        self.tweets_list.append(tweet)
        self.user_set.add(tweet.username)
        self.tweet_count += 1

    def get_user_count(self):
        return len(self.user_set)

    def set_bursty_score(self, score):
        self.bursty_score = score

    def __str__(self):
        burst_score = str(round(self.bursty_score, 2))
        return "Segment: " + self.segment + " , tweet_count: "+str(self.tweet_count)+" , user_count: "\
               +str(self.get_user_count()) + " , bursty score: " + str(burst_score)