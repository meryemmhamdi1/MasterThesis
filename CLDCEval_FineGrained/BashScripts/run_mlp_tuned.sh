
echo "Running Monolingual Experiments: English"
python ../main.py --mode="mono" --model-choice="mlp-tuned" --language="english"

echo "Running Monolingual Experiments: French"
python ../main.py --mode="mono" --model-choice="mlp-tuned" --language="french"

echo "Running Monolingual Experiments: German"
python ../main.py --mode="mono" --model-choice="mlp-tuned" --language="german"

echo "Running Monolingual Experiments: Italian"
python ../main.py --mode="mono" --model-choice="mlp-tuned" --language="italian"

echo "Running Multilingual Embeddings training and testing on All Languages: English, German, French, Italian"
python ../main.py --mode="multi" --model-choice="mlp-tuned" --multi-train="en,de,fr,it"

echo "Running Multilingual Embeddings on English only and testing on all languages: English, German, French, Italian"
python ../main.py --mode="multi" --model-choice="mlp-tuned"  --multi-train="en"


