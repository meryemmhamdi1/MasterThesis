#echo "Running MLP-Fine Monolingual Experiments: English"
#python ../main_new.py --mode="mono" --data-choice="churn" --model-choice="mlp-tuned" --language="english" --languages="english,german"

#echo "Running MLP-Fine Monolingual Experiments: German"
#python ../main_new.py --mode="mono" --data-choice="churn" --model-choice="mlp-tuned" --language="german" --languages="english,german"

echo "Running MLP-Fine Multilingual Embeddings training and testing on All Languages: English, German"
python ../main_new.py --multi-model-file="multiSkip_512_normalized" --mode="multi" --data-choice="churn" --model-choice="mlp-tuned" --multi-train="en,de" --languages="english,german"

#######################################################################################################################

#echo "Running CNN Monolingual Experiments: English"
#python ../main_new.py --mode="mono" --data-choice="churn" --model-choice="cnn" --language="english" --languages="english,german"

#echo "Running CNN Monolingual Experiments: German"
#python ../main_new.py --mode="mono" --data-choice="churn" --model-choice="cnn" --language="german" --languages="english,german"

echo "Running CNN Multilingual Embeddings training and testing on All Languages: English, German"
python ../main_new.py --multi-model-file="multiSkip_512_normalized" --mode="multi" --data-choice="churn" --model-choice="cnn" --multi-train="en,de" --languages="english,german"


#######################################################################################################################
#echo "Running GRU-ATT Monolingual Experiments: English"
#python ../main_new.py --mode="mono" --data-choice="churn" --model-choice="gru-att" --language="english" --languages="english,german"

#echo "Running GRU-ATT Monolingual Experiments: German"
#python ../main_new.py --mode="mono" --data-choice="churn" --model-choice="gru-att" --language="german" --languages="english,german"

echo "Running GRU-ATT Multilingual Embeddings training and testing on All Languages: English, German"
python ../main_new.py --multi-model-file="multiSkip_512_normalized" --mode="multi" --data-choice="churn" --model-choice="gru-att" --multi-train="en,de" --languages="english,german"

