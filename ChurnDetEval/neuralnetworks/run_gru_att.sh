#!/bin/sh
echo "Running Monolingual Experiments: English"
python run_model.py --network-type="gru_att" --train-mode="en_mono" --word-embeddings-path="/aimlx/MonolingualEmbeddings/GoogleNews-vectors-negative300.bin"

# Running Monolingual Experiments: French
#python run_model.py --network-type="gru_att"

echo "Running Monolingual Experiments: German"
python run_model.py --network-type="gru_att" --word-embeddings-path="/aimlx/MonolingualEmbeddings/german.model" --train-mode="de_mono"

echo "Running Monolingual Experiments: Italian"
python run_model.py --network-type="gru_att" --word-embeddings-path="/aimlx/MonolingualEmbeddings/wiki.it.vec" --train-mode="it_mono"

echo "Running Multilingual Embeddings on All Languages: English, German, French, Italian"
# Running Multilingual Embeddings on All Languages: English, German, French, Italian
python run_model.py --network-type="gru_att" --word-embeddings-path="/aimlx/MultilingualEmbeddings/multiSkip_40_normalized" --train-mode="en,de,fr,it"

