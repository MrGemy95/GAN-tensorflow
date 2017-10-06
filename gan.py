from config import GanConfig
from gan_model import GanModel
from data_generator import DataGenerator
from train import GanTrainer
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_boolean('is_train', True, """ Whether it is a training or testing""")
tf.app.flags.DEFINE_boolean('cont_train', False, """ whether to Load the Model and Continue Training or not """)


class GAN:
    def __init__(self, sess):
        """
        :param sess: the tensorflow session
        """
        self.sess = sess
        self.config = GanConfig()
        self.model = GanModel(self.config)
        self.data = DataGenerator(self.config)
        self.trainer = GanTrainer(self.sess, self.model, self.data, self.config)

    def train(self):
        self.trainer.train()


def main(_):
    init = tf.global_variables_initializer()
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.7
    sess = tf.Session(config=config)
    sess.run(init)
    gan = GAN(sess)

    if FLAGS.is_train:
        gan.train()


if __name__ == '__main__':
    tf.app.run()
