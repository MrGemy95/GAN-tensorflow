from base_train import BaseTrain
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import utils
from utils import inverse_transform
import scipy.misc


class GanTrainer(BaseTrain):
    def __init__(self, sess, model, data, config):
        super(GanTrainer, self).__init__(sess, model, data, config)

    def train(self):
        self.sess.run(tf.global_variables_initializer())

        self.sample_z = np.random.uniform(-1, 1, size=(self.config.batch_size, self.config.z_dim))  # training(
        for cur_epoch in range(self.cur_epoch_tensor.eval(self.sess), self.config.n_epochs + 1, 1):
            loop = tqdm(self.data.next_batch(), total=self.config.nit_epoch, desc="epoch-" + str(cur_epoch) + "-")
            cur_iterration = 0

            for batch_images in loop:
                batch_z = np.random.uniform(-1, 1, [self.config.batch_size, self.config.z_dim]).astype(np.float32)

                feed_dict = {self.model.inputs: batch_images, self.model.z: batch_z, self.model.is_training: True}
                # discriminator train step
                _, summary_str, discriminator_loss_real, discriminator_loss_fake, discriminator_loss_total = self.sess.run(
                    [self.model.discriminator_train_step, self.model.discriminator_summary, self.model.discriminator_loss_real,
                     self.model.discriminator_loss_fake, self.model.discriminator_loss],
                    feed_dict=feed_dict)

                self.summary_writer.add_summary(summary_str, cur_iterration)

                # train generator
                feed_dict = {self.model.z: batch_z, self.model.is_training: True}

                # generator train step
                _, summary_str, generator_loss = self.sess.run(
                    [self.model.generator_train_step, self.model.generator_summary, self.model.generator_loss],
                    feed_dict=feed_dict)

                self.summary_writer.add_summary(summary_str, cur_iterration)

                cur_iterration += 1
                if cur_iterration > self.config.nit_epoch:
                    break


            loop.close()
            # getting the current global step to add summary
            cur_it = self.global_step_tensor.eval(self.sess)

            # test generator every epoch
            feed_dict = {self.model.z: self.sample_z, self.model.is_training: False}
            generated_images = self.sess.run(
                [self.model.generated_output],
                feed_dict=feed_dict)

            summaries_dict = dict()
            summaries_dict['generated_images'] = inverse_transform((generated_images[0]))

            self.summarize(cur_it, summaries_dict=summaries_dict,
                           scope='test')
            for i in range(self.config.batch_size):
                scipy.misc.imsave('./res/'+str(cur_epoch)+'_'+str(i)+'.png', inverse_transform(np.squeeze(generated_images[0][i])))


            # increment_epoch
            self.cur_epoch_assign_op.eval(session=self.sess,
                                          feed_dict={self.cur_epoch_input: self.cur_epoch_tensor.eval(self.sess) + 1})
            # Save the current checkpoint
            if cur_epoch % self.config.save_every == 0:
                self.save()











