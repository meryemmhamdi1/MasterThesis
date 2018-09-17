# multilingual:

This software deliverable for this thesis consists of a python library for inducing and evaluating multilingual embeddings in different applications including their qualitative analysis.
The objectives of this library are to provide an end to end benchmarking system where given cross lingual datasets for document classification, churn detection or event detection and multilingual embeddings on any set of languages, it performs adapted preprocessing, trains/fine tunes multilingual models and return evaluation results against monolingual experiments.


# Installing Requirements:
We provide a Dockerfile to build and run an image that has all required packages for executing all programs in this library.

# Code Walkthrough:
Code for reproducing the results is organized into the following sub-directories:

## [MultiEmb]: Generation of Multilingual Embeddings

This sub-directory includes code for generating embeddings using offline methodologies: multi(pseudo_dict), multi(expert_dict) and multi(sem). It also contains code for training and evaluating Multi-Tasking methodology on CLDC.

### Offline Methodologies:
The main script to run main.py with option --dict-type= "expert" for generating embeddings using ground truth dictionaries and with option --dict-type="psdo" for generating embeddings with pseudo dictionary. Other options include what src languages --src-lang (separated by undescore) and which target language --trg-lang to use, whether to use dimensionality reduction or not in --dim-red and the path of monolingual embeddings to use in --mono-emb-path.

Run compute_alignments.py for applying already learned alignment directly on monolingual embeddings and for saving the whole in one file.

#### Translation Matrices and Visualization:
Run compute_precision_translation.py to get the precision @1 and @5 of the translation matrices on any source and target languages against a test dictionary.

#### Multi-Tasking:
Run main.py for multilingual experiments adapting --train-langs and --test-langs as needed. For monolingual experiments, the script main_mono.py is to be used.

### [CLDCEval]: Cross Lingual Document Classification using Multilingual Embeddings:
The first goal consists of Benchmarking different multilingual embedding models and their application to document classification across languages.

### Translation Matrices and Visualization:
Run compute_precision_translation.py to get the precision @1 and @5 of the translation matrices on any source and target languages against a test dictionary.

### Multi-Tasking:
Run main.py for multilingual experiments adapting --train-langs and --test-langs as needed. For monolingual experiments, the script main\_mono.py is to be used.

## [CLDCEval]: Cross Lingual Document Classification using Multilingual Embeddings

The first goal consists of benchmarking different multilingual embedding models and their application to document classification across languages.

The main script to run is "main.py" which after performing data preprocessing, converts each word to each corresponding vector using either monolingual or multilingual embedding model depending on the mode of evaluation chosen, then runs classification of choice including simple MLP, linear SVM and multi filter CNN. There is a separate script "main\_fine\_tune\_mlp.py" for running MLP in which embedding layer is trainable to further fine tune the embeddings to the current task.

Here are some examples of running "main_new.py":

    * Multilingual Experiment Training on English and testing on all languages: python main_new.py --mode="multi" --model-choice="mlp" --multi-train="en" --multi-model-file="multiCCA_512_normalized" --embed-dim=512
    * Multilingual Experiment Training and testing on all languages: python main_new.py --mode="multi" --model-choice="mlp-tuned" --multi-train="en,de,fr,it" --multi-model-file="multiSkip_40_normalized" --embed-dim= 40 --epochs=100
    * Monolingual English Experiment: python main_new.py --mode="mono" --model-choice="mlp-tuned" --language="english" --embed-dim=300
    * Monolingual Italian Experiment: python main_new.py --mode="mono" --model-choice="cnn" --language="italian" --embed-dim=300

To get the whole list and description of options with which the scripts can be executed, have a look at get_args.py script. The main flags to provide/change depend on the type of the experiment to be executed:
    * Monolingual/Multilingual model: mode="mono" / "multi"
        * If monolingual model is chosen, the flag language need to be specified. The currently supported languages are "english", "french", "italian", and "german". In this case, the program will only focus on that particular language by training, validating and testing on it. There is a default monolingual embedding model path for each language. If you want to try another gensim model, then go to --w2v-en, w2v-de, w2v-fr or w2v-it.
        * If multilingual model is chosen, flag multi-train need to be specified for example:
            * en: mean training and validating on english only and testing on all languages (english, german, italian, french)
            * fr: mean training and validating on french only and testing on all languages (english, german, italian, french)
            * de: mean training and validating on german only and testing on all languages (english, german, italian, french)
            * it: mean training and validating on italian only and testing on all languages (english, german, italian, french)
            * en, de: mean training and validating on english and german and testing on all languages (english, german, italian, french)
            * en,de,fr,it: mean training, validating and testing on all four languages
    * In case of multilingual embeddings, you can specify the directory and model name in --model-dir and --multi-model-file respectively. Don't forget to change --embed-dim accordingly.
    * Choice of Document Classification Model: model-choice="mlp" or "cnn" or "svm"
    * Choice of the dataset: it is by default rcv (Reuters) dataset. If you provide the raw dataset in the form of folders of xml files for each language, it will parse and preprocess it from scratch. Otherwise, if you the preprocessed version of x_train, x_dev, and x_test (where each split is a list of document saved in the form of pickle where each document element is a list of sentences in the document), you can skip the parsing and provide any dataset you want non-tokenized and non-indexed. You can further adapt it by skipping some steps from the pipeline and providing the preprocessed version directly.
    * Preparing the embeddings model: The embedding model should be a UTF-8 encoded plain text file where each line is a word. Each line begin with a lowercased surface form, prefixed by the 2-letter ISO 639-1 code of the language (e.g., "en:school" or "fr:école") followed by the floating-point values of each dimensions in the word embedding. All fields must be delimited with one space, and each line must end with a "\n" (as opposed to "\r\n". If the format of your embedding is different feel free to adapt function: load_multi_vectors in data_utils.


For Launching experiments with different training modes using a particular text classification architecture, you can use scripts in BashScripts. The analysis of the results is done separately using Jupyter Notebooks for better visualizations and enable easier debugging and plot generation. Please have a look at New Experiment Results.ipynb for the generation mechanism of the latest results.
## [CLDCEval_FineGrained]: Fined Grained Cross Lingual Document Classification:

The same code and methodology in CLDCEval is extended to include also Fine-Grained Multi-Label Dataset. In other words, this is a generalization of CLDCEval that works for fine-grained and coarse grained datasets.

## [ChurnDetEval]: Cross Lingual Churn Detection:
This contains code to benchmarking different multilingual embedding model as they are applied to churn detection across two languages: English and German. Code is taken and adapted from Maxime Coriou and Athanasios Giannakopoulos from their work: "Everything Matters: A Robust Ensemble Architecture for End-to-End Text Classification".

There are two ways to evaluate Churn Detection either with or without Cross Validation. In the former case, running script run_model_cross_val.py,  which expects one Tweet file and bot conversation file per language where each line represents a tweet or bot utterance delimited by its churn label, will go over the whole end-to-end pipeline: preprocess data (CrossValDataset.py), create 10 fold cross validation which creates 10 different training and testing splits (without validation), train models (NeuralNets.py with metrics defined in metrics_no_dev.py).
To change the default parameters used for running the scripts, adapt get_args.py as needed:

    * --word-embeddings-path: path to the monolingual or multilingual embeddings
    * --train-mode: the languages (separated by comma) on which the model is trained: en or de for monolingual training or en,de for multilingual training
    * --dataset= the choice of the dataset (default is churn)
    * --network-type and the hyperparameters related to it

Bash Scripts ready for running example experiments are also provided. We analyze the results using Jupyter Notebook: "Analyzing Cross Validation Churn Detection Results.ipynb"


## [EvDetEval]: Cross Lingual Event Detection
This is a library for training unsupervised multilingual event detection.
This source code consists of four modules:
    * WorldCupTwitterCollection: This is a streaming application using tweepy. The main script to run here is stream_tweets.py which is collects tweets based on hashtags. We create a Twitter application for each language which can be specified in --lang argument. Consumer key, consumer secret, access token and access token secret should be provided in the configuration file stream_config.yaml. There are different hashtags files: officialTags, teamTags1, teamTags2. The output for each hour of each particular day of collection is saved as a separate json file.
    * PreprocessingModule: There are three different scripts depending on the type of preprocessor: SpaCy (preprocess_sp_tweet.py), TreeTagger(preprocess_tt_tweets.py) or tokenization only(preprocess_tweet.py) which depends on the language. Prior to that, it is necessary to prepare the file by processing json hour files and converting them into one file for tweets in a particular day and language. This is done using prepare_csv_data.py
    * BurstySegmentExtraction: This module defines functionalities for splitting tweets into time windows and sub-windows, analysis of burstiness of unigrams in the tweets for each language alone and for the aggregation of many languages. Run detect_bursty_segments.py to get a feel of how it works where the input is the path to the Twitter dataset, the set of languages, start and end time of the time window and the output is the list of terms with their burstiness scores.
    * EventSegmentClustering: this module defines functions for converting bursty triggers to their embedding representation, computing semantic similarity using either embeddings or tf-idf and clustering using knn.
    * PostFiltering: This module applies only to multilingual extension choice 2 (clustering all triggers then selecting bursty event clusters). It computes burstiness of events based on the burstiness scores of its constituing triggers.

By running main.py, you can train an end to end system which executes all modules in the pipeline and gives back event clusters for different match sets. The inputs to the script are the following arguments:

    * --data-choice: path to the Twitter dataset (world cup 18 by default)
    * --round-choice: which part of the world cup to analyze. This can be either Group Round (1st), Group Round (2nd), Group Round(3rd), Round of 8, Quart-Finals, Semi-Finals, 3rd Place, Final. A fixture with metadata (start time and day, teams involved, round, times of goals, cards, etc) needs to be provided for each match in one csv file.
    * --trigger-mode: "mono" for multilingual trigger burstiness or "multi" for multilingual event burstiness (options 1 or 2 in multilingual extension respectively)
    * --sem-sim-mode: either word embeddings "emb" or "tf-idf".
    * --event-mode: either "mono" to perform event detection for each language independently among the languages of the involved teams in addition to English or "multi" for the aggregation of all languages using multilingual embeddings.
    * --time-window-size: default is 2 hours which is the duration of the game
    * --sub-window-size: default is 1 hour
    * --mono-model-dir, --multi-model-dir, --multi-model-file for directories and name of monolingual and multilingual embeddings files
    * --neighbours, --min-cluster-segments: number of minimum number of nearest neighbours and minimum number of triggers per clusters in the clustering algorithm
