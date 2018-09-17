
#echo "Running CNN Monolingual Experiments: English"
#python ../main.py --mode="mono" --data-choice="rcv" --model-choice="cnn" --language="english"

#echo "Running CNN Monolingual Experiments: French"
#python ../main.py --mode="mono" --data-choice="rcv" --model-choice="cnn" --language="french"

#echo "Running CNN Monolingual Experiments: German"
#python ../main.py --mode="mono" --data-choice="rcv" --model-choice="cnn" --language="german"

#echo "Running CNN Monolingual Experiments: Italian"
#python ../main.py --mode="mono" --data-choice="rcv" --model-choice="cnn" --language="italian"

#echo "Running CNN Multilingual Embeddings training and testing on All Languages: English, German, French, Italian"
#python ../main.py --mode="multi" --data-choice="rcv" --model-choice="cnn" --multi-train="en,de,fr,it"

####################################################################################################################

#echo "Running MLP-Tuned Monolingual Experiments: English"
#python ../main.py --mode="mono" --data-choice="rcv" --model-choice="mlp-tuned" --language="english"

#echo "Running MLP-Tuned Monolingual Experiments: French"
#python ../main.py --mode="mono" --data-choice="rcv" --model-choice="mlp-tuned" --language="french"

#echo "Running MLP-Tuned Monolingual Experiments: German"
#python ../main.py --mode="mono" --data-choice="rcv" --model-choice="mlp-tuned" --language="german"

#echo "Running MLP-Tuned Monolingual Experiments: Italian"
#python ../main.py --mode="mono" --data-choice="rcv" --model-choice="mlp-tuned" --language="italian"

#echo "Running MLP-Tuned Multilingual Embeddings training and testing on All Languages: English, German, French, Italian"
#python ../main.py --mode="multi" --data-choice="rcv" --model-choice="mlp-tuned" --multi-train="en,de,fr,it"

####################################################################################################################

#echo "Running GRU-ATT Monolingual Experiments: English"
#python ../main.py --mode="mono" --data-choice="rcv" --model-choice="gru-att" --language="english" --epochs=1

#echo "Running GRU-ATT Monolingual Experiments: French"
#python ../main.py --mode="mono" --data-choice="rcv" --model-choice="gru-att" --language="french" --epochs=1

#echo "Running GRU-ATT Monolingual Experiments: German"
#python ../main.py --mode="mono" --data-choice="rcv" --model-choice="gru-att" --language="german" --epochs=1

#echo "Running GRU-ATT Monolingual Experiments: Italian"
#python ../main.py --mode="mono" --data-choice="rcv" --model-choice="gru-att" --language="italian" --epochs=1

echo "Running GRU-ATT Multilingual Embeddings training and testing on All Languages: English, German, French, Italian"
python ../main.py --mode="multi" --data-choice="rcv" --model-choice="gru-att" --multi-train="en,de,fr,it" --epochs=1 --model-dir="/aimlx/Embeddings/MultilingualEmbeddings/"