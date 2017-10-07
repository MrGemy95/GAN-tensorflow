import tensorflow as tf


class BaseTrain:
    def __init__(self, sess, model, data, config):
        self.model = model
        self.config = config
        self.sess = sess
        self.data = data

        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(self.init)

        self.summary_placeholders = {}
        self.summary_ops = {}

        self.summary_writer = tf.summary.FileWriter(self.config.summary_dir, self.sess.graph)

        if self.config.load:
            sess.model.load(self.sess)

    def summarize(self, step, scope='train', summaries_dict=None, summaries_merged=None):
        """
        Add the summaries to tensorboard
        :param step:
        :param summaries_dict:
        :param summaries_merged:
        :return:
        """
        with tf.variable_scope(scope):

            if summaries_dict is not None:
                summary_list = []
                for tag, value in summaries_dict.items():
                    if tag not in self.summary_ops:
                        self.summary_placeholders[tag] = tf.placeholder('float32', value.shape, name=tag)
                        if len(value.shape) <= 1:
                            self.summary_ops[tag] = tf.summary.scalar(tag, self.summary_placeholders[tag])
                        else:
                            self.summary_ops[tag] = tf.summary.image(tag, self.summary_placeholders[tag])

                    summary_list.append(self.sess.run(self.summary_ops[tag], {self.summary_placeholders[tag]: value}))

                for summary in summary_list:
                    self.summary_writer.add_summary(summary, step)
                self.summary_writer.flush()
            if summaries_merged is not None:
                self.summary_writer.add_summary(summaries_merged, step)
                self.summary_writer.flush()
