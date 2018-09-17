# itemid, date, lang, texts, countries, industries, topics, lead_topic
class RCVDoc:
    def __init__(self, itemid, date, lang, texts, countries, industries, topics, lead_topic):
        self.itemID = itemid
        self.date = date
        self.lang = lang
        self.texts = texts
        self.countries = countries
        self.industries = industries
        self.topics = topics
        self.lead_topic = lead_topic

    def toString(self):
        return "item ID is = " + self.itemID + " on date = "+ self.date + " written in language = " + self.lang + \
                "list of sentences = " + self.texts + " countries = "+ self.countries + " industries = " + self.industries + \
                "topics = " + self.topics + " lead topic = "+ self.lead_topic

    def getItemID(self):
        return self.itemID

    def getDate(self):
        return self.date

    def getLanguage(self):
        return self.lang

    def getTexts(self):
        return self.texts

    def getCountries(self):
        return self.countries

    def getIndustries(self):
        return self.industries

    def getTopics(self):
        return self.topics

    def getLeadTopic(self):
        return self.lead_topic


    def setItemID(self, itemID):
        self.itemID = itemID

    def setDate(self, date):
        self.date = date

    def setLanguage(self, language):
        self.language = language

    def setTexts(self, texts):
        self.texts = texts

    def setCountries(self, countries):
        self.countries = countries

    def setIndustries(self, industries):
        self.industries = industries

    def setTopics(self, topics):
        self.topics = topics

    def setLeadTopic(self, lead_topic):
        self.lead_topic = lead_topic



