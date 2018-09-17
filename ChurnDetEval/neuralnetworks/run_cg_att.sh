#!/bin/sh
# Running Monolingual Experiments: English
#python run_model.py --train-mode="en_mono" --word-embeddings-path="/aimlx/MonolingualEmbeddings/GoogleNews-vectors-negative300.bin"

# Running Monolingual Experiments: French
#python run_model.py

# Running Monolingual Experiments: German
#python run_model.py --word-embeddings-path="/aimlx/MonolingualEmbeddings/german.model" --train-mode="de_mono"

# Running Monolingual Experiments: Italian
#python run_model.py --word-embeddings-path="/aimlx/MonolingualEmbeddings/wiki.it.vec" --train-mode="it_mono"

# Running Multilingual Embeddings on All Languages: English, German, French, Italian
python run_model.py --word-embeddings-path="/aimlx/MultilingualEmbeddings/multiSkip_40_normalized" --train-mode="en,de,fr,it"

