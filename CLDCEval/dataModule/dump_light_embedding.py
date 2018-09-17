import cPickle as pkl
from tqdm import tqdm

with open("/aimlx/en,de,fr,it_fasttext_en_de_fr_it.vec_vocab.p", "rb") as file:
    vocab = pkl.load(file)

print("Reading English")
with open("/aimlx/MUSE/dumped/debug/supervised_de_en/vectors-en.txt") as file_model:
    data = file_model.readlines()

model = {}
for i in tqdm(range(0, len(data[1:]))):
    word = data[i].split(" ")[0]
    vectors = data[i].split(" ")[1:]
    model.update({"en_"+word: vectors})

model_small = {}
for word in tqdm(vocab.keys()):
    if word in model:
        model_small.update({word: model[word]})

print("Reading french")
with open("/aimlx/MUSE/dumped/debug/supervised_fr_en/vectors-fr.txt") as file_model:
    data = file_model.readlines()

model = {}
for i in tqdm(range(0, len(data[1:]))):
    word = data[i].split(" ")[0]
    vectors = data[i].split(" ")[1:]
    model.update({"fr_"+word: vectors})

for word in tqdm(vocab.keys()):
    if word in model:
        model_small.update({word: model[word]})

print("Reading German")
with open("/aimlx/MUSE/dumped/debug/supervised_de_en/vectors-de.txt") as file_model:
    data = file_model.readlines()

model = {}
for i in tqdm(range(0, len(data[1:]))):
    word = data[i].split(" ")[0]
    vectors = data[i].split(" ")[1:]
    model.update({"de_"+word: vectors})

for word in tqdm(vocab.keys()):
    if word in model:
        model_small.update({word: model[word]})

print("Reading Italian")
with open("/aimlx/MUSE/dumped/debug/supervised_it_en/vectors-it.txt") as file_model:
    data = file_model.readlines()

model = {}
for i in tqdm(range(0, len(data[1:]))):
    word = data[i].split(" ")[0]
    vectors = data[i].split(" ")[1:]
    model.update({"it_"+word: vectors})

for word in tqdm(vocab.keys()):
    if word in model:
        model_small.update({word: model[word]})


print("Dumping smaller embeddings")
with open("/aimlx/supervised_fastext.txt", "w") as file:
    for word in model_small:
        file.write(word+ " "+ model_small[word] + "\n")


