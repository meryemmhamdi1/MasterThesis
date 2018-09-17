"""
Created on Wed Feb 28 2018

@author: Meryem M'hamdi (Meryem.Mhamdi@epfl.ch)

"""

class MultiLangVector:

    def __init__(self, language, word, vector):
        self.language = language
        self.word = word
        self.vector = vector

    def toString(self):
        return "Language = "+self.language + " Word= "+self.word + "Vector= "+self.vector

    def getLanguage(self):
        return self.language

    def setLanguage(self, language):
        self.language = language

    def getWord(self):
        return self.word

    def setWord(self, word):
        self.word = word

    def getVector(self):
        return self.vector

    def setVector(self, vector):
        self.vector = vector


