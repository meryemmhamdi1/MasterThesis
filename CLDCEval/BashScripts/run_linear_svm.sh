#!/bin/sh

data_choice = "rcv"
model-choice = "linear-svm"

echo "Running Monolingual Experiments: English"
python ../main_new.py --mode="mono" --data-choice= ${data_choice} --model-choice=${model_choice} --language="english"

echo "Running Monolingual Experiments: French"
python ../main_new.py --mode="mono" --data-choice= ${data_choice} --model-choice=${model_choice}--language="french"

echo "Running Monolingual Experiments: German"
python ../main_new.py --mode="mono" --data-choice= ${data_choice} --model-choice=${model_choice} --language="german"

echo "Running Monolingual Experiments: Italian"
python ../main_new.py --mode="mono" --data-choice= ${data_choice} --model-choice=${model_choice} --language="italian"

echo "Running Multilingual Embeddings on All Languages: English, German, French, Italian"
python ../main_new.py --mode="multi" --data-choice= ${data_choice} --model-choice=${model_choice} --multi-train="en,de,fr,it"

echo "Running Multilingual Embeddings on English only and testing on all languages: English, German, French, Italian"
python ../main_new.py --mode="multi" --data-choice= ${data_choice} --model-choice=${model_choice} --multi-train="en"


