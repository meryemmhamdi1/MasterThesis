import cPickle as pkl
import os

from Models.mlp_fine_tune_model import *
from Models.mlp_model import *
from Models.multi_filter_cnn_model import *
from Models.bi_gru_att_model import *

import vocab_embedding
from dataModule import data_utils
from dataModule.RCV import new_processor_no_dev
from get_args import *
from metrics_no_dev import *
import numpy as np
from keras.callbacks import ModelCheckpoint


#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#os.environ["THEANO_FLAGS"] = "/device:GPU:1"


def save_results():
    results_dict = {}
    # Train Results
    results_dict['y_train_pred'] = metrics.train_preds
    results_dict['y_train_trg'] = metrics.train_trgs
    results_dict['train_metrics'] = metrics.train_metrics

    # Test Results
    for lang in metrics.test_preds_dict:
        results_dict['y_test_pred_' + lang] = metrics.test_preds_dict[lang]
        results_dict['y_test_trg_' + lang] = metrics.test_trgs_dict[lang]
        results_dict['test_metrics_' + lang] = metrics.test_metrics_dict[lang]

    # Saving losses
    results_dict['train_loss'] = history.history['loss']
    # results_dict['val_loss'] = history.history['val_loss']

    with open(save_path + "_results.p", "wb") as dict_pkl:
        pkl.dump(results_dict, dict_pkl)

    # serialize weights to HDF5
    cldc_model.model.save(save_path + "_" + args.model_weights_path)

if __name__ == '__main__':

    global args, lang_list, lang_dict, model_lang_mapping, train_lang, test_lang_list

    args = get_args()

    lang_dict = {}
    with open("../../iso_lang_abbr.txt") as iso_lang:
        for line in iso_lang:
            lang_dict.update({line.split(":")[1][:-1]: line.split(":")[0]})

    model_results_dir = args.model_save_path + args.model_choice.upper() + "_Keras_Models_RCV/"
    if not os.path.isdir(model_results_dir):
        os.makedirs(model_results_dir)

    """ 2. Data Extraction """
    if args.mode == "mono":
        model_dir = args.w2v_dir
        model_lang_mapping = {'english': args.w2v_en, 'german': args.w2v_de, 'french': args.w2v_fr, 'italian': args.w2v_it}
        model_file = model_lang_mapping[args.language]
        save_path = model_results_dir + args.language + "_mono"
        train_lang = [lang_dict[args.language]]
        test_lang_list = [args.language]
    else:
        model_dir = args.model_dir
        model_file = args.multi_model_file
        save_path = model_results_dir + args.multi_train + "_" + args.multi_model_file
        train_lang = args.multi_train.split(",")
        test_lang_list = args.languages.split(',')

    """ 3. Embedding Loading """
    model, embed_dim = vocab_embedding.load_embeddings(args.mode, args.language, model_dir, model_file, lang_dict)

    """ I. Preprocessing """
    x_train_dict = {}
    y_train_dict = {}
    x_test_dict = {}
    y_test_dict = {}
    data_util_dict = {}

    for language in test_lang_list:
        print("Processing language=> ", language)
        data_util = data_utils.DataUtils(args.data_rcv, args.pre_dir, args.stop_pos_path, args.lemma_use,
                                         args.stop_use, language, model_dir, model_file, embed_dim)

        dp = new_processor_no_dev.RCVProcessorNoDev(data_util, lang_dict)

        x_train_dict.update({lang_dict[language]: dp.x_train_pro})
        y_train_dict.update({lang_dict[language]: dp.y_train})
        x_test_dict.update({lang_dict[language]: dp.x_test_pro})
        y_test_dict.update({lang_dict[language]: dp.y_test})
        data_util_dict.update({lang_dict[language]: data_util})

        n_classes = dp.n_classes

    print("3. Creation Global Vocabulary ...")
    x_train_all = []
    x_test_all = []
    for lang in x_train_dict:
        x_train_all += x_train_dict[lang]
        x_test_all += x_test_dict[lang]

    x_all = x_train_all + x_test_all
    vocab = vocab_embedding.create_vocabulary(x_all)
    max_sequences = max([len(doc) for doc in x_all])

    print("max_sequences=", max_sequences)

    print("4. Converting to ids and Padding to fixed length ...")
    sequences_train_dict = {}
    sequences_test_dict = {}
    for lang in x_train_dict:
        sequences_train, sequences_test = \
            vocab_embedding.convert_ids_no_dev(x_train_dict[lang], x_test_dict[lang], vocab)

        data_train, data_test = vocab_embedding.pad_fixed_length_no_dev(sequences_train, sequences_test,
                                                                        len(vocab), max_sequences)
        
        sequences_train_dict.update({lang: data_train})
        sequences_test_dict.update({lang: data_test})

    print("6. Building Embedding Matrix")
    embedding_matrix = vocab_embedding.build_embedding_matrix(vocab, model, embed_dim)

    if args.model_choice == "mlp":
        cldc_model = MLPModel(embed_dim, max_sequences, args.dense, args.dropout, args.learning_rate,
                              args.beta_1, args.beta_2, args.epsilon, n_classes, vocab, embedding_matrix)
    elif args.model_choice == "mlp-tuned":
        cldc_model = MLPFineTuneModel(embed_dim, max_sequences, args.dense, args.dropout, args.learning_rate,
                                      args.beta_1, args.beta_2, args.epsilon, n_classes, vocab, embedding_matrix)
    elif args.model_choice == "cnn":
        cldc_model = MultiFilterCNNModel(max_sequences, vocab, embed_dim, embedding_matrix, args.filter_sizes,
                                         args.num_filters, args.dropout, args.learning_rate,
                                         args.beta_1, args.beta_2, args.epsilon, n_classes)
    else:
        cldc_model = BiGRUAttModel(max_sequences, vocab, embed_dim, embedding_matrix, args.bidirectional,
                                   args.num_of_units, args.dropout, args.learning_rate, args.beta_1,
                                   args.beta_2, args.epsilon, n_classes)

    if len(train_lang) == 1:  ## Training and Validation on English and Testing on other languages
        print("Training and Validation on %s " % args.multi_train)
        x_train = sequences_train_dict[train_lang[0]]
        y_train = y_train_dict[train_lang[0]]
        x_test = {train_lang[0]: sequences_test_dict[train_lang[0]]}
        y_test = {train_lang[0]: y_test_dict[train_lang[0]]}

    else:  ## Training and validation on at least two languages and Testing on all other languages
        print("Training and Validation on %s " % args.multi_train)
        x_train = np.concatenate([sequences_train_dict[train_lang[i]] for i in range(0, len(train_lang))], axis=0)
        y_train = np.concatenate([y_train_dict[train_lang[i]] for i in range(0, len(train_lang))], axis=0)

    metrics = MetricsNoDev(x_train, y_train, sequences_test_dict, y_test_dict, args.mode, n_classes)

    """
    checkpoint = ModelCheckpoint('weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1,
                                 save_best_only=True, mode='auto')
    """

    history = cldc_model.model.fit(x_train, y_train,
                                   batch_size=args.batch_size,
                                   epochs=args.epochs,
                                   shuffle=True,
                                   callbacks=[metrics])#, checkpoint])#, EarlyStopping(monitor='val_loss', patience=0)])

    # Evaluate the model on training dataset
    scores_train = cldc_model.model.evaluate(x_train, y_train, verbose=0)
    print("%s: %.2f%%" % (cldc_model.model.metrics_names[1], scores_train[1] * 100))

    save_results()
