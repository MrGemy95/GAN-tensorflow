
import tensorflow as tf
from utils import leaky_relu
from base_model import BaseModel
class GanModel(BaseModel):
    def __init__(self,config):
        super(GanModel, self).__init__(config)
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.z = tf.placeholder(tf.float32, [None, self.config.z_dim], name='z')
        self.inputs = tf.placeholder(tf.float32, [None] + self.config.input_size, name='real_images')
        self.build_model()
        self.init_saver()


    def discriminator_templete(self, input):
        with tf.name_scope("discriminator_network"):
            h1 = tf.layers.conv2d(input, 64, (4, 4), (2, 2), name='conv1',
                                  kernel_initializer=tf.truncated_normal_initializer(stddev=.02))
            ################################################################################
            h2 = tf.layers.conv2d(h1, 128, (4, 4), (2, 2), name='conv2',
                                  kernel_initializer=tf.truncated_normal_initializer(stddev=.02))
            bn_h2 = leaky_relu(tf.layers.batch_normalization(h2,momentum=.9, epsilon=1e-5,training=self.is_training))
            flatten_h2 = tf.contrib.layers.flatten(bn_h2)
            ################################################################################
            h3 = tf.layers.dense(flatten_h2, 1024, name='fc1',
                                 kernel_initializer=tf.random_normal_initializer(stddev=.02))
            bn_h3 = leaky_relu(tf.layers.batch_normalization(h3,momentum=.9, epsilon=1e-5,training=self.is_training))
            ################################################################################

            logits = tf.layers.dense(bn_h3, 1, name='output',
                                     kernel_initializer=tf.random_normal_initializer(stddev=.02))
            out = tf.nn.sigmoid(logits)
            return out, logits

    def generator(self, z):
        with tf.variable_scope("generator_network"):
            h1 = tf.layers.dense(z, 1024, kernel_initializer=tf.random_normal_initializer(stddev=.02), name='fc1')
            bn_h1 = tf.nn.relu(tf.layers.batch_normalization(h1,momentum=.9, epsilon=1e-5,training=self.is_training))
            ################################################################################
            h2 = tf.layers.dense(bn_h1, 128 * 7 * 7, kernel_initializer=tf.random_normal_initializer(stddev=.02),
                                 name='fc2')
            bn_h2 = tf.nn.relu(tf.layers.batch_normalization(h2,momentum=.9, epsilon=1e-5,training=self.is_training))
            reshaped_h2 = tf.reshape(bn_h2, [-1, 7, 7, 128])
            ################################################################################
            h3 = tf.layers.conv2d_transpose(reshaped_h2, 64, kernel_size=(4, 4), strides=(2, 2), padding='SAME',
                                            kernel_initializer=tf.random_normal_initializer(stddev=.02), name='deconv1')
            bn_h3 = tf.nn.relu(tf.layers.batch_normalization(h3,momentum=.9, epsilon=1e-5,training=self.is_training))
            ################################################################################
            h4 = tf.layers.conv2d_transpose(bn_h3, 1, kernel_size=(4, 4), strides=(2, 2), padding='SAME',
                                            kernel_initializer=tf.random_normal_initializer(stddev=.02), name='deconv2')
            ################################################################################
            output = tf.nn.sigmoid(h4)

            return output

    def build_model(self):
        with tf.variable_scope('GAN_model'):
            self.discriminator = tf.make_template('discriminator', self.discriminator_templete)
            discriminator_real_output, discriminator_real_logits = self.discriminator(self.inputs)

            self.generated_output = self.generator(self.z)
            d_fake_output, d_fake_logits = self.discriminator(self.generated_output)

        with tf.name_scope('loss'):
            # discriminator
            self.discriminator_loss_real = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=discriminator_real_logits,
                                                        labels=tf.ones_like(discriminator_real_output)))
            self.discriminator_loss_fake = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_logits, labels=tf.zeros_like(d_fake_logits)))

            self.discriminator_loss = self.discriminator_loss_real + self.discriminator_loss_fake

            # generator loss
            self.generator_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_logits, labels=tf.ones_like(d_fake_logits)))

        # to backpropagate each network just to it's variables
        all_vars = tf.trainable_variables()
        discriminator_vars = [v for v in all_vars if 'discriminator' in v.name]
        generator_vars = [v for v in all_vars if 'generator' in v.name]

        with tf.name_scope('optimizers'):
            ex_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(ex_ops):
                self.discriminator_train_step = tf.train.AdamOptimizer(self.config.learning_rate,
                                                                       beta1=self.config.beta1).minimize(
                    self.discriminator_loss, var_list=discriminator_vars)

            ex_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            ex_ops = ex_ops[4:10]

            with tf.control_dependencies(ex_ops):
                self.generator_train_step = tf.train.AdamOptimizer(self.config.learning_rate * 2,
                                                                   beta1=self.config.beta1).minimize(
                    self.generator_loss, var_list=generator_vars)

        discriminator_loss_real = tf.summary.scalar("d_loss_real", self.discriminator_loss_real)
        discriminator_loss_fake = tf.summary.scalar("d_loss_fake", self.discriminator_loss_fake)
        discriminator_loss = tf.summary.scalar("d_loss", self.discriminator_loss)
        generator_loss = tf.summary.scalar("g_loss", self.generator_loss)

        self.generator_summary = tf.summary.merge([discriminator_loss_fake, generator_loss])
        self.discriminator_summary = tf.summary.merge([discriminator_loss_real, discriminator_loss])
