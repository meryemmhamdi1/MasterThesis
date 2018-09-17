import glob

if __name__ == '__main__':

    class_dict = {"C": 0, "E": 1, "G": 2, "M": 3}
    lang_dict = {"en": "english", "fr": "french", "it": "italian", "de": "german"}
    DATA_PATH = "/Users/MeryemMhamdi/EPFL/Spring2018/Thesis/3AlgorithmsImplementation/multilingual-embeddings" \
                "-eval-portal/eval-data/classification/reuters_da+de+en+es+fr+it+sv.dev/document-representations/" \
                "data/rcv-from-binod/train/"

    SAVE_PATH = "/Users/MeryemMhamdi/EPFL/Spring2018/Thesis/3AlgorithmsImplementation/KlementievSplit/"
    for lang in glob.glob(DATA_PATH + "*"):
        language = lang.split("/")[-1][:2]
        to_file_str = ''
        if lang.split("/")[-1][:2] in ["en", "fr", "de", "it"]:
            for directory in glob.glob(lang + "*"):
                class_ = class_dict[directory.split("/")[-1]]
                for xml_file in glob.glob(directory + "*"):
                    with open(xml_file) as file:
                        to_file_str += file.readlines() + "\t" + class_ + "\n"

        print("Saving to train file")
        with open(SAVE_PATH+"/"+lang_dict[language]+"/train.txt", "w") as file:
            file.write(to_file_str)

