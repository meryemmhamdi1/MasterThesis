class Tweet():
    def __init__(self, tweet_id, date, hour, minutes, text, username):
        self.tweet_id = tweet_id
        self.date = date
        self.hour = hour
        self.minutes = minutes
        self.text = text
        self.username = username

    def __str__(self):
        return "Tweet with id: "+self.tweet_id+" created on: "+self.date + " at: "+self.hour + ":" + self.minutes +\
               " , text: " + self.text + " , by username: " + self.username

