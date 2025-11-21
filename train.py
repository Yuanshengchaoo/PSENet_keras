import os
import tensorflow as tf
from keras import callbacks, optimizers
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import matplotlib.pyplot as plt

plt.switch_backend('agg')

from global_var import myModelConfig
from model import PSENet
from data_generator import pasi_data
from loss_metric import score_loss, siam_loss, locate_loss, score_metric, locate_metric
from self_callbacks import MyEarlyStop

global patient_dict

os.environ["CUDA_VISIBLE_DEVICES"] = myModelConfig.availiable_gpus
steps_per_epoch_train = int(myModelConfig.num_train_examples_per_epoch // myModelConfig.batch_size)
steps_per_epoch_val = int(myModelConfig.num_val_examples_per_epoch // myModelConfig.batch_size)

data_loader = pasi_data()
train_gen = data_loader.train_generator(myModelConfig.batch_size)
valid_gen = data_loader.valid_generator(myModelConfig.batch_size)

strategy = tf.distribute.MirroredStrategy() if myModelConfig.num_gpus > 1 else tf.distribute.get_strategy()
with strategy.scope():
    siamese_model = PSENet(myModelConfig)

    print(siamese_model.input)
    print(siamese_model.output)
    siamese_model.summary()

    optimizer = optimizers.SGD(learning_rate=myModelConfig.learning_rate, momentum=myModelConfig.momentum, nesterov=True)

    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=myModelConfig.learning_rate_decay_factor, patience=5,
                                            verbose=1, mode='min', cooldown=10, min_lr=0.00001)
    my_early_stop = MyEarlyStop(myModelConfig.checkpoint_dir)
myTensorboard = callbacks.TensorBoard(log_dir=myModelConfig.summary_dir, histogram_freq=0, write_graph=False,
                                      write_images=True)

my_call_back = [my_early_stop, reduce_lr, myTensorboard]
siamese_model.compile(
    loss={"scoreA": score_loss, "scoreB": score_loss, "scoreSiam": siam_loss,
          "single_model_1": locate_loss,
          "single_model_2": locate_loss,
          "single_model_3": locate_loss,
          "single_model_4": locate_loss,
          "single_model_5": locate_loss,
          "single_model_6": locate_loss,
          "single_model_7": locate_loss,
          "single_model_8": locate_loss,
          "single_model_9": locate_loss,
          "single_model_10": locate_loss},
    loss_weights={"scoreA": 0.1, "scoreB": 0.1, "scoreSiam": 0.05,
                  "single_model_1": 1,
                  "single_model_2": 1,
                  "single_model_3": 1,
                  "single_model_4": 1,
                  "single_model_5": 1,
                  "single_model_6": 1,
                  "single_model_7": 1,
                  "single_model_8": 1,
                  "single_model_9": 1,
                  "single_model_10": 1},
    optimizer=optimizer,
    metrics={"scoreA": score_metric, "scoreB": score_metric, "scoreSiam": score_metric,
             "single_model_1": locate_metric,
             "single_model_2": locate_metric,
             "single_model_3": locate_metric,
             "single_model_4": locate_metric,
             "single_model_5": locate_metric,
             "single_model_6": locate_metric,
             "single_model_7": locate_metric,
             "single_model_8": locate_metric,
             "single_model_9": locate_metric,
             "single_model_10": locate_metric,})

 print("compiled")

 history = None
 fit_kwargs = dict(
     x=train_gen,
     epochs=myModelConfig.num_epochs,
     verbose=1,
     steps_per_epoch=steps_per_epoch_train,
     callbacks=my_call_back,
     validation_data=valid_gen,
     validation_steps=steps_per_epoch_val,
     initial_epoch=0,
 )

 # Guard against legacy kwargs that were removed in TF 2.17/Keras 3.4
 for legacy_kwarg in ("use_multiprocessing", "workers", "max_queue_size"):
     fit_kwargs.pop(legacy_kwarg, None)

 try:
     history = siamese_model.fit(**fit_kwargs)

 except KeyboardInterrupt:
     print("Early stop by user !")

 except StopIteration:
     print("Training process finished !")

 except TypeError as exc:
     # If a user-modified script reintroduces deprecated kwargs, drop them and retry once
     legacy_triggers = ("use_multiprocessing", "workers", "max_queue_size")
     if any(trigger in str(exc) for trigger in legacy_triggers):
         for kw in legacy_triggers:
             fit_kwargs.pop(kw, None)
         history = siamese_model.fit(**fit_kwargs)
     else:
         print("training process error")
         raise


finally:
    if history:
        def plot_train_history(history, train_metrics, val_metrics):
            plt.plot(history.history.get(train_metrics), '-o')
            plt.plot(history.history.get(val_metrics), '-o')
            plt.ylabel(train_metrics)
            plt.xlabel('Epochs')
            plt.legend(['train', 'validation'])


        plt.figure(figsize=(8, 4))
        plt.subplot(2, 2, 1)
        plot_train_history(history, 'loss', 'val_loss')
        plt.subplot(2, 2, 2)
        plot_train_history(history, 'scoreA_score_metric', 'val_scoreA_score_metric')
        plt.subplot(2, 2, 3)
        plot_train_history(history, 'scoreB_score_metric', 'val_scoreB_score_metric')
        plt.subplot(2, 2, 4)
        plot_train_history(history, 'scoreSiam_score_metric', 'val_scoreSiam_score_metric')

        plt.savefig(myModelConfig.history_file)
