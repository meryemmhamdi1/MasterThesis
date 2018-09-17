# encoding=utf8
import sys

reload(sys)
sys.setdefaultencoding("utf8")
import tensorflow as tf
from data_util import *
from multi_task_mono import *
#import cPickle as pkl

try:
    from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple, GRUCell
except ImportError:
    LSTMCell = tf.nn.rnn_cell.LSTMCell
    LSTMStateTuple = tf.nn.rnn_cell.LSTMStateTuple
    GRUCell = tf.nn.rnn_cell.GRUCell

import argparse
import numpy as np
import operator
import collections
from sklearn.metrics import f1_score, recall_score, precision_score
import json

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#os.environ["THEANO_FLAGS"] = "/job:localhost/replica:0/task:0/device:GPU:0"  # "/job:localhost/replica:0/task:0/device:GPU:1"#"/device:GPU:1"


def get_args():
    """
    Command line arguments

    Arguments set the default values of command line arguments
    :return:
    """
    parser = argparse.ArgumentParser()

    """ Dataset Path Parameters """
    parser.add_argument("--rcv-path", "-dp", type=str, default="/aimlx/Datasets/RCV/MultiLabels/")
    parser.add_argument("--europarl-path", "-ep", type=str, default="/aimlx/Datasets/EuroParl/")
    parser.add_argument("--model-dir", "-md", type=str, default="/aimlx/Embeddings/MonolingualEmbeddings/")
    parser.add_argument("--model-file", "-mf", type=str, default="expert_dict_dim_red_en_de_fr_it.txt")
    parser.add_argument("--train-langs", "-trl", type=str,
                        default="english")  # default="english,french,german,italian")
    parser.add_argument("--test-langs", "-tl", type=str, default="english")
    parser.add_argument("--mode", "-m", type=str, default="single-label")

    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()
    n_epochs = 10
    n_mini_batches = 15
    # Read data for aligned sentences, documents
    docs_train, y_train, docs_dev, y_dev, docs_test, y_test = read_cldc_docs(args.rcv_path, args.train_langs,
                                                                             args.test_langs)

    # Read all batches for sentences_src and sentences_trg and document separately.
    vocab_dict = {}
    doc_train_split = {}
    doc_dev_split = {}
    doc_test_split = {}
    num = 10000

    for lang in docs_train:
        print("Preprocessing train docs for language: " + lang)
        train_split, vocab_dict = split_docs(docs_train[lang][:num], vocab_dict, lang)
        doc_train_split.update({lang: train_split})
        print("Preprocessing dev docs for language: " + lang)
        dev_split, vocab_dict = split_docs(docs_dev[lang][:num], vocab_dict, lang)
        doc_dev_split.update({lang: dev_split})

    for lang in docs_test:
        print("Preprocessing test docs for language: " + lang)
        test_split, vocab_dict = split_docs(docs_test[lang][:num], vocab_dict, lang)
        doc_test_split.update({lang: test_split})

    # sorted_dict = OrderedDict(sorted(vocab_dict.items(), key=lambda x: x[1]))
    print("len(vocab_dict):", len(vocab_dict))
    new_dict = dict(sorted(vocab_dict.iteritems(), key=operator.itemgetter(1), reverse=True)[:3000000])

    vocab_list = sorted(new_dict)
    with open("/aimlx/Results/MultiTask/"+args.train_langs+"_vocab.p", "w") as file:
        for word in vocab_list:
            file.write(word + "\n")
    vocab = dict([x, y] for (y, x) in enumerate(vocab_list))

    # Load embeddings
    print(vocab_list[0:10])
    model, embed_dim = load_fast_text(args.model_dir, args.test_langs, vocab_list)

    embedding_matrix = build_embedding_matrix(vocab, model, embed_dim, vocab_dict)
    if "," in args.test_langs:
        test_langs = args.test_langs.split(",")
    else:
        test_langs = [args.test_langs]
    tf.reset_default_graph()
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    #config.gpu_options.allow_growth = True
    with tf.Session(config=config) as session:
        model = MultiTaskEmbeddingsHANClassifier(
            vocab_size=len(vocab),
            embedding_size=300,
            classes=4,
            word_cell=GRUCell(10),
            sentence_cell=GRUCell(10),
            word_output_size=10,
            sentence_output_size=10,
            max_grad_norm=5.0,
            dropout_keep_proba=0.5,
        )
        global_step = tf.Variable(0, name="global_step", trainable=False)
        session.run(tf.global_variables_initializer())

        Y2_op = model.train_task2_op

        """Train/Dev/Test Doc splits"""
        doc_train = []
        label_train = []
        doc_dev = []
        label_dev = []
        for lang in docs_train:
            doc_train += doc_train_split[lang]
            doc_dev += doc_dev_split[lang]
            label_train += y_train[lang][:num]
            print("Class distribution TRAIN: ", collections.Counter(y_train[lang][:num]))
            label_dev += y_dev[lang][:num]
            print("Class distribution DEV: ", collections.Counter(y_dev[lang][:num]))

        doc_test = {}
        label_test = {}
        for lang in docs_test:
            doc_test.update({lang: doc_test_split[lang]})
            label_test.update({lang: y_test[lang][:num]})
            print("Class distribution TEST: ", collections.Counter(y_test[lang][:num]))

        batches_doc = doc_batch(doc_train, n_mini_batches, label_train)
        print("len(batches_doc):", len(batches_doc))
        print("n_mini_batches:", n_mini_batches)

        # Test batches
        batches_doc_test = {}
        for lang in docs_test:
            n_mini_bat_doc_test = n_mini_batches # int(len(doc_test[lang]) / int(len(doc_train) / n_mini_batches))
            batches_doc_test.update({lang: doc_batch(doc_test[lang], n_mini_bat_doc_test, label_test[lang])})

        test_acc_doc_lang = {}
        test_f1_score_doc_lang = {}
        test_recall_doc_lang = {}
        test_precision_doc_lang = {}
        test_true_doc_lang = {}
        test_pred_doc_lang = {}
        for lang in test_langs:
            test_acc_doc_lang.update({lang: []})
            test_f1_score_doc_lang.update({lang: []})
            test_recall_doc_lang.update({lang: []})
            test_precision_doc_lang.update({lang: []})
            test_true_doc_lang.update({lang: []})
            test_pred_doc_lang.update({lang: []})

        train_acc_doc = []
        train_f1_score_doc = []
        train_recall_doc = []
        train_precision_doc = []
        train_loss_doc = []
        train_true_labels = []
        train_pred_labels = []
        train_acc_avg = []
        train_f1_score_avg = []
        train_precision_avg = []
        train_recall_avg = []
        for epoch in range(n_epochs):
            train_loss_sub = []
            train_acc_sub = []
            train_f1_score_sub = []
            train_recall_sub = []
            train_precision_sub = []
            train_true_sub = []
            train_pred_sub = []
            for i in range(len(batches_doc)):  # number of minibatches
                # fd needs to change at each iteration
                fd = {
                    model.is_training: True,
                    model.inputs_doc: batches_doc[i]["docs"],
                    model.labels: batches_doc[i]["labels"],
                    model.sample_weights: batches_doc[i]["weights"],
                    model.sentence_lengths_doc: batches_doc[i]["document_sizes"],
                    model.word_lengths_doc: batches_doc[i]["sentence_sizes"],
                    model.embedding_placeholder: embedding_matrix,
                }
                loss_summary_task2 = tf.summary.scalar("loss2", model.loss)
                acc_summary_task2 = tf.summary.scalar("accuracy2", model.accuracy2)
                train_summary_op_2 = tf.summary.merge([loss_summary_task2, acc_summary_task2])
                if np.random.rand() < 0.5:  # self.cosine, self.mismatch_loss, self.loss_match, self.loss, self.total_labels,
                    _, step, summaries_2, Y2_loss, accuracy2, prediction = \
                        session.run([Y2_op, global_step, train_summary_op_2, model.loss, model.accuracy2,
                                     model.prediction], fd)
                    train_acc_sub.append(accuracy2)
                    recall = recall_score(batches_doc[i]["labels"], prediction, average='macro')
                    precision = precision_score(batches_doc[i]["labels"], prediction, average='macro')
                    if recall * precision == 0:
                        f1 = 0
                    else:
                        f1 = 2 * (recall * precision) / (recall + precision)
                    train_f1_score_sub.append(f1)
                    train_recall_sub.append(recall)
                    train_precision_sub.append(precision)
                    train_loss_sub.append(Y2_loss)
                    train_true_sub += list(batches_doc[i]["labels"])
                    train_pred_sub += list(prediction)
                    if i % 10 == 0:
                        print("accuracy2= ", accuracy2, "Precision:", precision, "Recall:", recall, "F1_score:", f1,
                              " Y2_loss:", Y2_loss)
                if i == len(batches_doc) - 1:
                    train_acc_doc.append(train_acc_sub)
                    train_f1_score_doc.append(train_f1_score_sub)
                    train_precision_doc.append(train_precision_sub)
                    train_recall_doc.append(train_recall_sub)

                    train_true_labels.append(train_true_sub)
                    train_pred_labels.append(train_pred_sub)

                    mean_train_acc = np.mean(train_acc_sub)
                    train_acc_avg.append(mean_train_acc)
                    mean_train_f1 = np.mean(train_f1_score_sub)
                    train_f1_score_avg.append(mean_train_f1)
                    mean_train_precision = np.mean(train_precision_sub)
                    train_precision_avg.append(mean_train_precision)
                    mean_train_recall = np.mean(train_recall_sub)
                    train_recall_avg.append(mean_train_recall)
                    train_loss_doc.append(train_loss_sub)

                    print("MEAN TRAIN Metrics: => accuracy:", mean_train_acc, " f1 score:", mean_train_f1,
                          "precision:", mean_train_precision, "recall:", mean_train_recall)
                    print("Computing Test Metrics >>>>>>")
                    for lang in test_langs:
                        accuracies_doc = []
                        f1_scores_doc = []
                        precisions_doc = []
                        recalls_doc = []
                        labels_true = []
                        labels_pred = []
                        for j in range(len(batches_doc_test)):
                            fd = {
                                model.is_training: False,
                                model.inputs_doc: batches_doc_test[lang][j]["docs"],
                                model.labels: batches_doc_test[lang][j]["labels"],
                                model.sample_weights: batches_doc_test[lang][j]["weights"],
                                model.sentence_lengths_doc: batches_doc_test[lang][j]["document_sizes"],
                                model.word_lengths_doc: batches_doc_test[lang][j]["sentence_sizes"],
                                model.embedding_placeholder: embedding_matrix,
                            }
                            accuracy2, prediction = \
                                session.run([model.accuracy2, model.prediction], fd)
                            accuracies_doc.append(accuracy2)
                            recall = recall_score(batches_doc_test[lang][j]["labels"], prediction, average='macro')
                            precision = precision_score(batches_doc_test[lang][j]["labels"], prediction, average='macro')
                            if recall * precision == 0:
                                f1 = 0
                            else:
                                f1 = 2 * (recall * precision) / (recall + precision) #f1_score(batches_doc[i]["labels"], prediction, average='macro')
                            f1_scores_doc.append(f1)
                            precisions_doc.append(precision)
                            recalls_doc.append(recall)
                            labels_true += list(batches_doc_test[lang][j]["labels"])
                            labels_pred += list(prediction)

                        test_acc_doc_lang[lang].append(np.mean(accuracies_doc)) 
                        test_f1_score_doc_lang[lang].append(np.mean(f1_scores_doc)) 
                        test_precision_doc_lang[lang].append(np.mean(precisions_doc)) 
                        test_recall_doc_lang[lang].append(np.mean(recalls_doc))
                        test_true_doc_lang[lang].append(labels_true)
                        test_pred_doc_lang[lang].append(labels_pred)

                        print("lang: ", lang, " Doc accuracy is: ", np.mean(accuracies_doc), 
                              " f1_score is: ", np.mean(f1_scores_doc), " precision is:", np.mean(precisions_doc), 
                              " recall is:", np.mean(recalls_doc))

            # Saving results in pickle file
            metrics = {"train_acc_doc": train_acc_doc, "train_f1_score_doc": train_f1_score_doc,
                       "train_precision_doc": train_precision_doc, "train_recall_doc": train_precision_doc,
                       "train_loss_doc": train_loss_doc, "train_true_doc": train_true_labels,
                       "train_pred_doc": train_pred_labels}
            for lang in test_acc_doc_lang:
                metrics["test_acc_doc_" + lang] = test_acc_doc_lang[lang]
                metrics["test_f1_score_doc_" + lang] = test_f1_score_doc_lang[lang]
                metrics["test_precision_doc_" + lang] = test_precision_doc_lang[lang]
                metrics["test_recall_doc_" + lang] = test_recall_doc_lang[lang]
                metrics["test_true_doc_" + lang] = test_true_doc_lang[lang]
                metrics["test_pred_doc_" + lang] = test_pred_doc_lang[lang]

            print("Writing the results .....")
            with open("/aimlx/Results/MultiTask/Tuned/"+args.train_langs+"_MONO_noemb_multi-tasking_epoch_"+str(epoch)+".txt", "w") as results_file:
                results_file.write(str(metrics))
                # results_file.write(json.dumps(metrics))
                # pkl.dump(metrics, results_file)
                # print(session.run(model.logits, fd))
                # session.run(model.train_op, fd)
