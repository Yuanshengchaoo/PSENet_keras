from keras import callbacks
import os
import numpy as np


class MyEarlyStop(callbacks.Callback):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.best_mae = float("inf")
        self.best_area = float("inf")
        self.best_ery = float("inf")
        self.best_sca = float("inf")
        self.best_ind = float("inf")
        self.best_pasi = float("inf")

        os.makedirs(self.path, exist_ok=True)


    def on_epoch_end(self, epoch, logs=None):
        print(logs)
        scoreA = logs.get('val_scoreA_score_metric')
        scoreB = logs.get("val_scoreB_score_metric")

        mae = (scoreA + scoreB) / 2.0

        global EPOCH
        EPOCH = epoch

        if mae < self.best_mae:
            print("\nValidation mae decreased from {} to {}, saving model".format(self.best_mae, mae))
            print("\nsave model to " + os.path.join(self.path, "epoch=" + str(epoch) + "_" + "mae=" + str(round(mae, 4)) + '.h5'))
            filelist = os.listdir(self.path)
            h5list = [elem for elem in filelist if os.path.splitext(elem)[-1] == '.h5']
            for elem in h5list:
                if "mae" in elem:
                    os.remove(os.path.join(self.path, elem))
            self.model.save(os.path.join(self.path, "epoch=" + str(epoch) + "_" + "mae=" + str(round(mae, 4)) + '.h5'),
                            overwrite=True)
            self.best_mae = mae

        else:
            print("\nValidation have not improvement, current average is {}, best average is {}".format(mae, self.best_mae))