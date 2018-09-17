# -*- coding: utf-8 -*-
import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


class Metrics(Callback):
    def __init__(self, x_train, one_hot_train, x_val, one_hot_dev, x_test_dict, one_hot_test_dict, mode, n_classes, single_label):
        # Train
        self.x_train = x_train
        self.one_hot_train = one_hot_train

        # Val
        self.x_val = x_val
        self.one_hot_dev = one_hot_dev

        # Test
        self.mode = mode

        self.x_test_dict = x_test_dict
        self.one_hot_test_dict = one_hot_test_dict

        self.n_classes = n_classes
        self.single_label = single_label
        self.threshold = 0.50

    def on_train_begin(self, logs={}):
        self.train_metrics = []
        self.train_preds = []
        self.train_trgs = []

        self.val_metrics = []
        self.val_preds = []
        self.val_trgs = []

        self.test_metrics_dict = {}
        self.test_preds_dict = {}
        self.test_trgs_dict = {}

    def on_epoch_end(self, epoch, logs={}):
        # Computing Training Metrics
        if self.single_label:
            predict = (np.asarray(self.model.predict(self.x_train))).argmax(1)
            targ = self.one_hot_train.argmax(1)
        else:
            y_pred = (np.asarray(self.model.predict(self.x_train)))
            predict = np.zeros((len(y_pred), self.n_classes))
            for i in range(0, len(y_pred)):
                for j in range(0, len(y_pred[i])):
                    if y_pred[i][j] > self.threshold:
                        predict[i][j] = 1

            targ = np.array(self.one_hot_train)

        _acc = accuracy_score(targ, predict)
        _f1_M, _f1_m = f1_score(targ, predict, average='macro'), f1_score(targ, predict, average='micro')
        _recall_M, _recall_m = recall_score(targ, predict, average='macro'), recall_score(targ, predict,
                                                                                          average='micro')
        _precision_M, _precision_m = precision_score(targ, predict, average='macro'), precision_score(targ, predict,
                                                                                                      average='micro')

        self.train_metrics.append({"acc": _acc, "f1_macro": _f1_M, "f1_micro": _f1_m,
                                   "recall_macro": _recall_M, "recall_micro": _recall_m,
                                   "precision_macro": _precision_M, "precision_micro": _precision_m})
        self.train_preds.append(predict)
        self.train_trgs.append(targ)

        print(" -- train_acc: %f — Macro train_f1: %f — micro train_f1: %f — Macro train_precision: %f "
               "— micro train_precision: %f — Macro train_recall %f — micro train_recall %f" %
               (_acc, _f1_M, _f1_m, _precision_M, _precision_m, _recall_M, _recall_m))

        # Computing Validation Metrics
        if self.single_label:
            predict = (np.asarray(self.model.predict(self.x_val))).argmax(1)
            targ = self.one_hot_dev.argmax(1)
            print("predict:", list(predict[:50]))
            print("targ:", list(targ[:50]))
        else:
            y_pred = (np.asarray(self.model.predict(self.x_val)))
            predict = np.zeros((len(y_pred), self.n_classes))
            for i in range(0, len(y_pred)):
                for j in range(0, len(y_pred[i])):
                    if y_pred[i][j] > self.threshold:
                        predict[i][j] = 1

            targ = np.array(self.one_hot_dev)


        _acc = accuracy_score(targ, predict)
        _f1_M, _f1_m = f1_score(targ, predict, average='macro'), f1_score(targ, predict, average='micro')
        _recall_M, _recall_m = recall_score(targ, predict, average='macro'), recall_score(targ, predict,
                                                                                          average='micro')
        _precision_M, _precision_m = precision_score(targ, predict, average='macro'), precision_score(targ, predict,
                                                                                                      average='micro')

        self.val_metrics.append({"acc": _acc, "f1_macro": _f1_M, "f1_micro": _f1_m,
                                 "recall_macro": _recall_M, "recall_micro": _recall_m,
                                 "precision_macro": _precision_M, "precision_micro": _precision_m})
        self.val_preds.append(predict)
        self.val_trgs.append(targ)

        print(" -- val_acc: %f — Macro val_f1: %f — micro val_f1: %f — Macro val_precision: %f "
               "— micro val_precision: %f — Macro val_recall %f — micro val_recall %f" %
               (_acc, _f1_M, _f1_m, _precision_M, _precision_m, _recall_M, _recall_m))

        # Computing Testing Metrics on each language
        for lang in self.x_test_dict:
            if self.single_label:
                predict = (np.asarray(self.model.predict(self.x_test_dict[lang]))).argmax(1)
                targ = self.one_hot_test_dict[lang].argmax(1)
            else:
                y_pred = (np.asarray(self.model.predict(self.x_test_dict[lang])))
                targ = np.array(self.one_hot_test_dict[lang])

                predict = np.zeros((len(y_pred), self.n_classes))
                for i in range(0, len(y_pred)):
                    for j in range(0, len(y_pred[i])):
                        if y_pred[i][j] > self.threshold:
                            predict[i][j] = 1


            _acc = accuracy_score(targ, predict)
            _f1_M, _f1_m = f1_score(targ, predict, average='macro'), f1_score(targ, predict, average='micro')
            _recall_M, _recall_m = recall_score(targ, predict, average='macro'), recall_score(targ, predict,
                                                                                              average='micro')
            _precision_M, _precision_m = precision_score(targ, predict, average='macro'), precision_score(targ,
                                                                                                          predict,
                                                                                                          average='micro')

            if lang in self.test_metrics_dict:
                metrics_list = self.test_metrics_dict[lang]
                metrics_list.append({"acc": _acc, "f1_macro": _f1_M, "f1_micro": _f1_m,
                                     "recall_macro": _recall_M, "recall_micro": _recall_m,
                                     "precision_macro": _precision_M, "precision_micro": _precision_m})

            else:
                metrics_list = [{"acc": _acc, "f1_macro": _f1_M, "f1_micro": _f1_m,
                                 "recall_macro": _recall_M, "recall_micro": _recall_m,
                                 "precision_macro": _precision_M, "precision_micro": _precision_m}]

            self.test_metrics_dict.update({lang: metrics_list})

            if lang in self.test_preds_dict:
                pred_list = self.test_preds_dict[lang]
                pred_list.append(predict)
            else:
                pred_list = [predict]

            self.test_preds_dict.update({lang: pred_list})

            if lang in self.test_trgs_dict:
                targ_list = self.test_trgs_dict[lang]
                targ_list.append(targ)
            else:
                targ_list = [targ]
            self.test_trgs_dict.update({lang: targ_list})

            print(" -- %s test_acc: %f — Macro test_f1: %f — micro test_f1: %f — Macro test_precision: %f "
                   "— micro val_precision: %f — Macro val_recall %f — micro val_recall %f" %
                   (lang.upper(), _acc, _f1_M, _f1_m, _precision_M, _precision_m, _recall_M, _recall_m))

        return
