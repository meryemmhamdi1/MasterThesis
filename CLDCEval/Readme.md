# Cross Language Document Classification (CLDC):

This is a downstream task for the evaluation of our multilingual embeddings trained either from scratch or reused from previous publications using different variations and adaptations.
This can use any of the three datasets:
* Deutsche Well Dataset: contains nearly 600,000 documents in 8 languages which are annotated by journalists with general and specific topic categories.
https://github.com/idiap/mhan
* Reuters Multilingual Dataset where the languages include in addition to English: Dutch, French, German, Chinese, Japanese, Russian, Portuguese, Spanish, Latin American Spanish, Italian, Danish, Norwegian, and Swedish.
We will restrict our evaluations to four languages:  English, French, German, Italian.
* TED Corpus: It is easily possible to re-create our CLDC corpus given the original WIT3 data. We used the following keywords as our classifier labels: technology, culture, science, global issues, design, business, entertainment, arts, politics, education, art, health, creativity, economics, biology.
Given the WIT3 data, development data is provided and we split the WIT3 training data into a new training/test split by considering all talks with a talk-id >= 1500 as test data. For reasons of keeping datasets separate and clean, we do not consider the WIT3 test data in this task.
http://www.clg.ox.ac.uk/tedcldc.html

In order to regenerate the results, run main.py which performs data preprocessing, trains and evaluates different models.
