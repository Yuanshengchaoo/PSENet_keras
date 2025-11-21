import os
import tensorflow as tf


class Logger:
    def __init__(self, config):
        self.config = config
        self.train_summary_writer = tf.summary.create_file_writer(os.path.join(self.config.summary_dir, "train"))
        self.test_summary_writer = tf.summary.create_file_writer(os.path.join(self.config.summary_dir, "test"))

    # it can summarize scalars and images.
    def summarize(self, step, summarizer="train", scope="", summaries_dict=None):
        """
        :param step: the step of the summary
        :param summarizer: use the train summary writer or the test one
        :param scope: variable scope (kept for compatibility)
        :param summaries_dict: the dict of the summaries values (tag,value)
        :return:
        """
        writer = self.train_summary_writer if summarizer == "train" else self.test_summary_writer

        if summaries_dict is None:
            return

        with writer.as_default():
            for tag, value in summaries_dict.items():
                if len(value.shape) <= 1:
                    tf.summary.scalar(tag, value if len(value.shape) == 0 else value[0], step=step)
                else:
                    tf.summary.image(tag, value, step=step)
            writer.flush()
