
echo "Running Monolingual Experiments: English"
python ../main_new.py --mode="mono" --data-choice="rcv-bal" --model-choice="gru-att" --language="english"

echo "Running Monolingual Experiments: French"
python ../main_new.py --mode="mono" --data-choice="rcv-bal" --model-choice="gru-att" --language="french"

echo "Running Monolingual Experiments: German"
python ../main_new.py --mode="mono" --data-choice="rcv-bal" --model-choice="gru-att" --language="german"

echo "Running Monolingual Experiments: Italian"
python ../main_new.py --mode="mono" --data-choice="rcv-bal" --model-choice="gru-att" --language="italian"

echo "Running Multilingual Embeddings training and testing on All Languages: English, German, French, Italian"
python ../main_new.py --mode="multi" --data-choice="rcv-bal" --model-choice="gru-att" --multi-train="en,de,fr,it"

echo "Running Multilingual Embeddings on English only and testing on all languages: English, German, French, Italian"
python ../main_new.py --mode="multi" --data-choice="rcv-bal" --model-choice="gru-att"  --multi-train="en"
