import os
import csv
import os
import numpy as np
import tensorflow as tf
from keras.models import load_model
from data_generator import test_generator
from loss_metric import score_loss, siam_loss, locate_loss, score_metric, locate_metric
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

os.environ["CUDA_VISIBLE_DEVICES"] = ""

model_path = "./experiments/checkpoints"

model_list = [elem for elem in os.listdir(model_path) if elem.endswith("h5")]
if len(model_list) > 1:
    print("find multiple models in {}".format(model_path))
    raise ValueError
if len(model_list) < 1:
    print("find no model in {}".format(model_path))
    raise ValueError

model_path = os.path.join(model_path, model_list[0])
print(model_path)
save_dir = "./experiments/results"
data_generator = test_generator()

model = load_model(model_path, custom_objects={"tf": tf, "score_loss": score_loss,
                                               "siam_loss": siam_loss, "locate_loss": locate_loss,
                                               "score_metric": score_metric, "locate_metric": locate_metric})
count = 0
index_name = ["area", "ery", "sca", "ind", "pasi"]
error = np.array([0, 0, 0, 0, 0], dtype=float)


def result_writer(f, img_name, predict_list, label_list):
    row = [img2_name]
    row.extend(predict_list)
    row.extend(label_list)
    f.writerow(row)


try:
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, "predict.csv"), "w", newline="") as f:
        csv_writer = csv.writer(f)
        head = ["name", "area", "erythema", "scale", "induration", "pasi", "area_label", "erythema_label",
                "scale_label", "induration_label", "pasi_label"]

        csv_writer.writerow(head)

        while True:
            img1_name, img2_name, img_list, label_list = next(data_generator)
            output = model.predict(img_list, batch_size=1, verbose=0, steps=None)

            img1_result = [float(elem) for elem in output[0].flatten()]
            label1_list = [float(elem) for elem in label_list[0][0]]

            img2_result = [float(elem) for elem in output[1].flatten()]
            label2_list = [float(elem) for elem in label_list[1][0]]

            siam_result = [float(elem) for elem in output[2].flatten()]
            label_siam_list = [float(elem) for elem in label_list[2][0]]

            count += 2
            error += np.abs(np.array(img1_result, dtype=float) - np.array(label1_list, dtype=float))
            error += np.abs(np.array(img2_result, dtype=float) - np.array(label2_list, dtype=float))

            # area_error += abs(label1_list[0] - img1_result[0])
            # area_error += abs(label2_list[0] - img2_result[0])
            #
            # ery_error += abs(label1_list[1] - img1_result[1])
            # ery_error += abs(label2_list[1] - img2_result[1])
            #
            # scale_error += abs(label1_list[2] - img1_result[2])
            # scale_error += abs(label2_list[2] - img2_result[2])
            #
            # ind_error += abs(label1_list[3] - img1_result[3])
            # ind_error += abs(label2_list[3] - img2_result[3])
            #
            # pasi_error += abs(label1_list[4] - img1_result[4])
            # pasi_error += abs(label2_list[4] - img2_result[4])

            result_writer(csv_writer, img1_name, img1_result, label1_list)
            result_writer(csv_writer, img2_name, img2_result, label2_list)
            result_writer(csv_writer, img1_name + "," + img2_name, siam_result, label_siam_list)

            print(img1_name, img2_name)

except StopIteration:

    if count != 0:
        error /= count

        for i in range(len(index_name)):
            print("mae of {} = {}".format(index_name[i], error[i]))

    print("{} img have been predicted".format(count))
    print("Done")
