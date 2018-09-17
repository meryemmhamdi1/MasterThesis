# This callback takes care of keeping track of the epochs and perform early stopping if necessary
import numpy as np
from keras.callbacks import Callback


class EarlyStoppingByPatience(Callback):
    def __init__(self, x_test, y_test, patience, id_to_label, batch_size):
        super(Callback, self).__init__()
        self.x_test = x_test
        self.y_test = y_test
        self.id_to_label = id_to_label
        self.batch_size = batch_size
        self.patience = patience

        self.max_acc = 0
        self.max_fscore = 0

        self.max_epoch_id = 0
        self.patience_counter = 0
        self.epochs = []
        self.epoch_counter = 1

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

            current_acc = numCorrLabels / float(numLabels)

            self.epochs.append({'Acc': float(current_acc)})

            if current_acc <= self.max_acc:
                self.patience_counter = self.patience_counter + 1
                if self.patience_counter >= self.patience:
                    self.model.stop_training = True
            else:
                self.max_acc = current_acc

                self.max_epoch_id = self.epoch_counter
                self.patience_counter = 0

            # Uncomment to write model to file
            # print("Saving model...")
            #self.model.save("/aimlx/ECGA_Keras_Models_RCV/en_" + str(current_acc) +'.h5')  # creates a HDF5 file 'my_model.h5'

            self.epoch_counter += 1

        # Compute F-score for task C
        def f_score(self, predictions):

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

            if precision + recall == 0:
                current_fscore = 0
            else:
                current_fscore = float(2 * precision * recall) / (precision + recall)

            print("\nValidation F-score: ", current_fscore, "\n")

            self.epochs.append({'fscore': float(current_fscore)})

            if current_fscore <= self.max_fscore:
                self.patience_counter = self.patience_counter + 1
                if self.patience_counter >= self.patience:
                    self.model.stop_training = True
            else:
                self.max_fscore = current_fscore

                self.max_epoch_id = self.epoch_counter
                self.patience_counter = 0

            # write model file
            # print("Saving model...")
            #self.model.save("/aimlx/ECGA_Keras_Models_RCV/en_" + str(current_fscore) +'.h5')  # creates a HDF5 file 'my_model.h5'

            self.epoch_counter += 1

        # Predict on validation set
        predictions = self.model.predict(self.x_test, batch_size=self.batch_size)

        # Select class with maximum probability
        y_hat = []
        for i, prediction in enumerate(predictions):
            y_hat.append(np.argmax(prediction))

        predictions = y_hat
        acc(self, predictions)  # Replace by: "fscore(predictions)" for Task C
