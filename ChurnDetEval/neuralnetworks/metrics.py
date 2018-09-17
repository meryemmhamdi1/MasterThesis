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

class Metrics(Callback):
    def __init__(self, x_train, one_hot_train, x_val, one_hot_dev, x_test_dict, one_hot_test_dict, mode,
                 n_classes, id_to_label, batch_size):
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
        self.id_to_label = id_to_label
        self.batch_size = batch_size

        print("New class***********************************************************")

    def f_score(self, targets, predictions):
        tpp = 0
        tpf = 0
        tpv = 0

        fnp = 0
        fnf = 0
        fnv = 0

        fpp = 0
        fpf = 0
        fpv = 0

        for i in range(len(predictions)):
            pred_label = self.id_to_label[predictions[i]]
            true_label = self.id_to_label[targets[i]]

            if true_label == "policy":
                if pred_label == "policy":
                    tpp += 1
                if pred_label == "fact":
                    fnp += 1
                    fpf += 1
                if pred_label == "value":
                    fnp += 1
                    fpv += 1
            if true_label == "fact":
                if pred_label == "policy":
                    fnf += 1
                    fpp += 1
                if pred_label == "fact":
                    tpf += 1
                if pred_label == "value":
                    fnf += 1
                    fpv += 1
            if true_label == "value":
                if pred_label == "policy":
                    fnv += 1
                    fpp += 1
                if pred_label == "fact":
                    fpf += 1
                    fnv += 1
                if pred_label == "value":
                    tpv += 1

        if tpp != 0:
            precisionp = float(tpp) / (tpp + fpp)
            recallp = float(tpp) / (tpp + fnp)
        else:
            precisionp = 0
            recallp = 0
        if tpf != 0:
            precisionf = float(tpf) / (tpf + fpf)
            recallf = float(tpf) / (tpf + fnf)
        else:
            precisionf = 0
            recallf = 0
        if tpv != 0:
            precisionv = float(tpv) / (tpv + fpv)
            recallv = float(tpv) / (tpv + fnv)
        else:
            precisionv = 0
            recallv = 0

        precision = (precisionp + precisionf + precisionv) / 3
        recall = (recallp + recallf + recallv) / 3

        if precision + recall == 0:
            current_fscore = 0
        else:
            current_fscore = float(2 * precision * recall) / (precision + recall)

        return precision, recall, float(current_fscore)

    def acc(self, targets, predictions):

        numLabels = 0
        numCorrLabels = 0

        for i in range(len(predictions)):
            pred_label = self.id_to_label[predictions[i]]
            true_label = self.id_to_label[targets[i]]
            numLabels += 1
            if pred_label == true_label:
                numCorrLabels += 1

        current_acc = numCorrLabels / float(numLabels)

        return float(current_acc)

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
        predictions = self.model.predict(self.x_train, batch_size=self.batch_size)
        y_hat = []
        for i, prediction in enumerate(predictions):
            y_hat.append(np.argmax(prediction))

        predictions = y_hat

        targets = self.one_hot_train
        y_hat = []
        for i, targ in enumerate(targets):
            y_hat.append(np.argmax(targ))

        targets = y_hat

        _acc = self.acc(self, targets, predictions)
        _precision_M, _recall_M, _f1_M = self.acc(self, targets, predictions)

        self.train_metrics.append({"acc": _acc, "f1_macro": _f1_M, "precision_macro": _precision_M,
                                   "recall_macro": _recall_M})
        self.train_preds.append(predictions)
        self.train_trgs.append(targets)

        print(" -- train_acc: %f — Macro train_f1: %f — Macro train_precision: %f  — Macro train_recall %f " %
               (_acc, _f1_M, _precision_M, _recall_M))

        # Computing Validation Metrics
        predictions = self.model.predict(self.x_val, batch_size=self.batch_size)
        y_hat = []
        for i, prediction in enumerate(predictions):
            y_hat.append(np.argmax(prediction))

        predictions = y_hat

        targets = self.one_hot_val
        y_hat = []
        for i, targ in enumerate(targets):
            y_hat.append(np.argmax(targ))

        targets = y_hat

        _acc = self.acc(self, targets, predictions)
        _precision_M, _recall_M, _f1_M = self.acc(self, targets, predictions)

        self.val_metrics.append({"acc": _acc, "f1_macro": _f1_M, "precision_macro": _precision_M,
                                   "recall_macro": _recall_M})
        self.val_preds.append(predictions)
        self.val_trgs.append(targets)

        print(" -- val_acc: %f — Macro val_f1: %f — Macro val_precision: %f  — Macro val_recall %f " %
              (_acc, _f1_M, _precision_M, _recall_M))

        # Computing Testing Metrics on each language
        for lang in self.x_test_dict:
            ##########
            predictions = self.model.predict(self.x_test_dict[lang], batch_size=self.batch_size)
            y_hat = []
            for i, prediction in enumerate(predictions):
                y_hat.append(np.argmax(prediction))

            predictions = y_hat

            targets = self.one_hot_test_dict[lang]
            y_hat = []
            for i, targ in enumerate(targets):
                y_hat.append(np.argmax(targ))

            targets = y_hat

            _acc = self.acc(self, targets, predictions)
            _precision_M, _recall_M, _f1_M = self.acc(self, targets, predictions)


            #######
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
                pred_list.append(predictions)
            else:
                pred_list = [predictions]

            self.test_preds_dict.update({lang: pred_list})

            if lang in self.test_trgs_dict:
                targ_list = self.test_trgs_dict[lang]
                targ_list.append(targets)
            else:
                targ_list = [targets]
            self.test_trgs_dict.update({lang: targ_list})

            print(" -- %s test_acc: %f — Macro test_f1: %f — Macro test_precision: %f  — Macro test_recall %f " %
                (lang.upper(), _acc, _f1_M, _precision_M, _recall_M))

        return
