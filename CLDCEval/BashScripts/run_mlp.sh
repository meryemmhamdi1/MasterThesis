#echo "Running Monolingual Experiments: English"
#python ../main_new.py --mode="mono" --data-choice="rcv-bal" --model-choice="mlp" --language="english"

#echo "Running Monolingual Experiments: French"
#python ../main_new.py --mode="mono" --data-choice="rcv-bal" --model-choice="mlp" --language="french"

#echo "Running Monolingual Experiments: German"
#python ../main_new.py --mode="mono" --data-choice="rcv-bal" --model-choice="mlp" --language="german"

#echo "Running Monolingual Experiments: Italian"
#python ../main_new.py --mode="mono" --data-choice="rcv-bal" --model-choice="mlp" --language="italian"

echo "Running Multilingual Embeddings training and testing on All Languages: English, German, French, Italian"
python ../main_new.py --mode="multi" --data-choice="rcv" --model-choice="mlp" --multi-train="en,de,fr,it"

#echo "Running Multilingual Embeddings on English only and testing on all languages: English, German, French, Italian"
#python ../main_new.py --mode="multi" --data-choice="rcv-bal" --model-choice="mlp"  --multi-train="en"
