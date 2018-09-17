# This callback evaluate on the test set and write the predictions to a file. 

import keras
import numpy as np


class TestCallback(keras.callbacks.Callback):
    def __init__(self, x_test, y_test, id_to_label, batch_size):
        self.x_test = x_test
        self.y_test = y_test
        self.id_to_label = id_to_label
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs={}):

        # Compute accuracy
        def acc(self, predictions):
            numLabels = 0
            numCorrLabels = 0

            for i in range(len(predictions)):
                pred_label = self.id_to_label[predictions[i]]
                true_label = self.id_to_label[np.where(self.y_test[i] == 1)[0][0]]
                numLabels += 1
                if pred_label == true_label:
                    numCorrLabels += 1

            accuracy = numCorrLabels / float(numLabels)
            print("\nTest accuracy: ", accuracy, "\n")

        # Compute F-score
        def fscore(self, predictions):
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
                true_label = self.id_to_label[np.where(self.y_test[i] == 1)[0][0]]

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

            current_fscore = float(2 * precision * recall) / (precision + recall)

            print("\nTest F-score: ", current_fscore, "\n")

        predictions = self.model.predict(self.x_test, batch_size=self.batch_size)
        # parse predictions
        self.__parse_predictions(predictions)
        # evaluate performance
        y_hat = []
        for i, prediction in enumerate(predictions):
            y_hat.append(np.argmax(prediction))

        predictions = y_hat
        acc(self, predictions)  # Replace by: "fscore(self, predictions)" for Task C

    # Write predictions to file
    def __parse_predictions(self, predictions):
        y_hat = []
        for i, prediction in enumerate(predictions):
            y_hat.append(np.argmax(prediction))

        predictions = y_hat

        open("output/predictions_all.txt", 'w').close()
        for i in range(len(predictions)):
            with open("output/predictions_all.txt", "a") as f:
                pred_label = self.id_to_label[predictions[i]]
                true_label = self.id_to_label[np.where(self.y_test[i] == 1)[0][0]]
                f.write(str(i + 1) + " " + true_label + " " + pred_label + "\n")
