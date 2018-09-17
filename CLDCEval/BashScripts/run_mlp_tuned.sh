
#echo "Running Monolingual Experiments: English"
#python ../main_new.py --mode="mono" --data-choice="rcv" --model-choice="mlp-tuned" --language="english"

#echo "Running Monolingual Experiments: French"
#python ../main_new.py --mode="mono" --data-choice="rcv" --model-choice="mlp-tuned" --language="french"

#echo "Running Monolingual Experiments: German"
#python ../main_new.py --mode="mono" --data-choice="rcv" --model-choice="mlp-tuned" --language="german"

#echo "Running Monolingual Experiments: Italian"
#python ../main_new.py --mode="mono" --data-choice="rcv" --model-choice="mlp-tuned" --language="italian"

echo "MLP-TUNED Running Multilingual Embeddings training and testing on All Languages: English, German, French, Italian"
python ../main_new.py --data-choice="rcv" --mode="multi" --model-choice="cnn" --multi-train="en,de,fr,it" --multi-model-file="multiCCA_40_normalized"

#echo "CNN Running Multilingual Embeddings training and testing on All Languages: English, German, French, Italian"
#python ../main_new.py --data-choice="rcv" --mode="multi" --model-choice="cnn" --multi-train="en,de,fr,it" --multi-model-file="multiSkip_40_normalized"


#echo "CNN Running Multilingual Embeddings training and testing on All Languages: English, German, French, Italian SEM Embeddings"
#python ../main_new.py --data-choice="rcv" --mode="multi" --model-choice="cnn" --multi-train="en,de,fr,it" --multi-model-file="multi_embed_linear_projection"

#echo "CNN Running Multilingual Embeddings training and testing on All Languages: English, German, French, Italian SENT_ALI Embeddings"
#python ../main_new.py --data-choice="rcv" --mode="multi" --model-choice="cnn" --multi-train="en,de,fr,it" --multi-model-file="multilingual_embeddings_joint_en_de_fr_it.txt"

#echo "Running Multilingual Embeddings on English only and testing on all languages: English, German, French, Italian"
#python ../main_new.py --mode="multi" --data-choice="rcv" --model-choice="mlp-tuned"  --multi-train="en"


