# -*- coding: utf-8 -*-
import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

def tp_fp_fn(y_pred, y_true, n_classes):
    tp_dict = {}
    fp_dict = {}
    fn_dict = {}
    for class_n in range(1,len(n_classes)+1):
        tp = 0
        fp = 0
        fn = 0
        for i in range(0, len(y_true)):
            if y_true[i] == class_n and y_pred[i] == class_n:# True Positive
                tp = tp + 1
            if y_true[i] != class_n and y_pred[i] == class_n:# True Negative
                fp = fp + 1
            if y_true[i] == class_n and y_pred[i] != class_n:# False Negative
                fn = fn + 1
        tp_dict.update({class_n: tp})
        fp_dict.update({class_n: fp})
        fn_dict.update({class_n: fn})
    return tp_dict, fp_dict, fn_dict


def micro_precision(tp_dict, fp_dict, n_classes):
    mi_prec_num = 0
    mi_prec_deno = 0
    for class_n in range(1, len(n_classes)+1):
        mi_prec_num += tp_dict[class_n]
        mi_prec_deno += tp_dict[class_n] + fp_dict[class_n]

    return mi_prec_num/mi_prec_deno


def micro_recall(tp_dict, fn_dict, n_classes):
    mi_recall_num = 0
    mi_recall_deno = 0
    for class_n in range(1, len(n_classes)+1):
        mi_recall_num += tp_dict[class_n]
        mi_recall_deno += tp_dict[class_n] + fn_dict[class_n]

    return mi_recall_num/mi_recall_deno


def f1_score_mean(precision, recall):
    if precision * recall != 0:
        harm_mean = 2 * (precision*recall)/(precision+recall)
    else:
        harm_mean = 0
    return harm_mean


def macro_precision(tp_dict, fp_dict, n_classes):
    ma_prec = 0
    for class_n in range(1, len(n_classes)+1):
        ma_prec += (tp_dict[class_n] /(tp_dict[class_n]+fp_dict[class_n]))
    return ma_prec


def macro_recall(tp_dict, fn_dict, n_classes):
    ma_prec = 0
    for class_n in range(1, len(n_classes)+1):
        ma_prec += (tp_dict[class_n] /(tp_dict[class_n]+fn_dict[class_n]))
    return ma_prec

class MetricsNoDev(Callback):
    def __init__(self, x_train, one_hot_train, x_test_dict, one_hot_test_dict, x_bot_test, y_bot_test, mode, n_classes):
        # Train
        self.x_train = x_train
        self.one_hot_train = one_hot_train

        # Test
        self.mode = mode

        self.x_test_dict = x_test_dict
        self.one_hot_test_dict = one_hot_test_dict

        self.x_bot_test = x_bot_test
        self.y_bot_test = y_bot_test

        self.n_classes = n_classes

    def on_train_begin(self, logs={}):
        self.train_metrics = []
        self.train_preds = []
        self.train_trgs = []

        self.test_metrics_dict = {}
        self.test_preds_dict = {}
        self.test_trgs_dict = {}

        self.test_bot_metrics_dict = {}
        self.test_bot_preds_dict = {}
        self.test_bot_trgs_dict = {}

    def on_epoch_end(self, epoch, logs={}):
        # Computing Training Metrics
        predict = (np.asarray(self.model.predict(self.x_train))).argmax(1)
        targ = self.one_hot_train.argmax(1)

        _acc = accuracy_score(targ, predict)
        _f1_M = f1_score(targ, predict, average='macro')
        _recall_M = recall_score(targ, predict, average='macro')
        _precision_M = precision_score(targ, predict, average='macro')

        self.train_metrics.append({"acc": _acc, "f1_macro": _f1_M, "recall_macro": _recall_M, "precision_macro": _precision_M})
        self.train_preds.append(predict)
        self.train_trgs.append(targ)

        print(" -- train_acc: %f — Macro train_f1: %f — Macro train_precision: %f "
              " — Macro train_recall %f \n" %
              (_acc, _f1_M, _precision_M, _recall_M))


        # Computing Testing Metrics on each language
        for lang in self.x_test_dict:
            predict = (np.asarray(self.model.predict(self.x_test_dict[lang]))).argmax(1)
            targ = self.one_hot_test_dict[lang].argmax(1)

            _acc = accuracy_score(targ, predict)
            _f1_M = f1_score(targ, predict, average='macro')
            _recall_M = recall_score(targ, predict, average='macro')
            _precision_M = precision_score(targ, predict, average='macro')

            if lang in self.test_metrics_dict:
                metrics_list = self.test_metrics_dict[lang]
                metrics_list.append({"acc": _acc, "f1_macro": _f1_M, "recall_macro": _recall_M,
                                     "precision_macro": _precision_M})

            else:
                metrics_list = [{"acc": _acc, "f1_macro": _f1_M, "recall_macro": _recall_M,
                                 "precision_macro": _precision_M}]

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

            print(" -- %s test_acc: %f — Macro test_f1: %f — Macro test_precision: %f "
                  " — Macro test_recall %f \n" %
                  (lang.upper(), _acc, _f1_M, _precision_M, _recall_M))

        # Computing Bot Testing Metrics on each language
        for lang in self.x_bot_test:
            predict = (np.asarray(self.model.predict(self.x_bot_test[lang]))).argmax(1)
            targ = self.y_bot_test[lang].argmax(1)

            """
            if lang == "en":
                print(list(predict))

                print(list(targ))
                print(list(self.y_bot_test[lang]))
                
            """
            _acc = accuracy_score(targ, predict)
            _f1_M = f1_score(targ, predict, average='macro')
            _recall_M = recall_score(targ, predict, average='macro')
            _precision_M = precision_score(targ, predict, average='macro')

            if lang in self.test_bot_metrics_dict:
                metrics_list = self.test_bot_metrics_dict[lang]
                metrics_list.append({"acc": _acc, "f1_macro": _f1_M, "recall_macro": _recall_M,
                                     "precision_macro": _precision_M})

            else:
                metrics_list = [{"acc": _acc, "f1_macro": _f1_M, "recall_macro": _recall_M,
                                 "precision_macro": _precision_M}]

            self.test_bot_metrics_dict.update({lang: metrics_list})

            if lang in self.test_bot_preds_dict:
                pred_list = self.test_bot_preds_dict[lang]
                pred_list.append(predict)
            else:
                pred_list = [predict]

            self.test_bot_preds_dict.update({lang: pred_list})

            if lang in self.test_bot_trgs_dict:
                targ_list = self.test_bot_trgs_dict[lang]
                targ_list.append(targ)
            else:
                targ_list = [targ]
            self.test_bot_trgs_dict.update({lang: targ_list})

            print(" BOT EVALUATION -- %s test_acc: %f — Macro test_f1: %f — Macro test_precision: %f "
                  " — Macro test_recall %f \n" %
                  (lang.upper(), _acc, _f1_M, _precision_M, _recall_M))

        return
