import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def weight_variable(name, shape):
    return tf.get_variable(name, shape=shape,
       initializer=tf.contrib.layers.xavier_initializer())
def bias_variable(name, shape):
    return tf.get_variable(name, shape=shape,
       initializer=tf.constant_initializer(0.))

class VariationalAutoEncoder(object):

    def __init__(self,
            lr=1e-3,
            batch_size=100,
            latent_dim=15,
            dim_image=784,
            hidden_dim=500,
            saved_path=None,
            saving_path = None,
            test_flag = None,
            num_epochs=50,
            model_dir=None):# For giving the location where the path needs to be saved
        self.learning_rate = lr
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.dim_image = dim_image
        self.hidden_dim = hidden_dim
        self.saved_path = saved_path
        self.saving_path = saving_path
        self.model_dir = model_dir
        # Build all graph
        self.x = tf.placeholder(tf.float32, shape=[None, self.dim_image])
        self.buildEncoder()
        self.latent = self.generate_latent()
        self.reconstruct = self.buildDecoder(self.latent)
        self.loss()
        self.train = tf.train.AdamOptimizer(learning_rate=self.learning_rate)\
            .minimize(self.total_loss)

        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        if (test_flag is None) and (self.saving_path is None):
            self.saving_path = 'default_model_hd{}_ld{}_e{}'.format(self.hidden_dim, self.latent_dim, num_epochs)
            print("Model saving path not given. Hence saving to default location:",self.saving_path)
        if self.saved_path is not None:
            self.saver.restore(self.sess, self.saved_path)
            print("Restored model")

    # Build encoder
    def buildEncoder(self):
        W1 = weight_variable("we1", [self.dim_image, self.hidden_dim])
        b1 = bias_variable("be1", [self.hidden_dim])
        h = tf.nn.tanh(tf.matmul(self.x, W1) + b1)
        W2 = weight_variable("we2", [self.hidden_dim, 2 * self.latent_dim])
        b2 = bias_variable("be2", [2 * self.latent_dim])
        z = tf.matmul(h, W2) + b2
        #print ("latent shape", z.get_shape())
        self.mu = z[:, :self.latent_dim]
        self.sigma = tf.exp(z[:, self.latent_dim:])

    # Generate a latent variable sampled from gaussian distritbution
    def generate_latent(self):
        latent = self.mu + self.sigma * tf.random_normal(
            tf.shape(self.mu), 0, 1, dtype=tf.float32)
        return latent

    # Decodes a latent variable and returns the reconstructed image.
    def buildDecoder(self, latent):
        W1 = weight_variable("wd1", [self.latent_dim, self.hidden_dim])
        b1 = bias_variable("bd1", [self.hidden_dim])
        reconstruct = tf.nn.tanh(tf.matmul(latent, W1) + b1)
        W2 = weight_variable("wd2", [self.hidden_dim, self.dim_image])
        b2 = bias_variable("bd2", [self.dim_image])
        reconstruct = tf.nn.sigmoid(tf.matmul(reconstruct, W2) + b2)
        return reconstruct

    # Given initial data, reconstructed data and parameters, compute the loss.
    def loss(self):
        self.likelihood = -tf.reduce_mean(tf.reduce_sum(
            self.x * tf.log(self.reconstruct) +
            (1 - self.x) * tf.log(1 - self.reconstruct), 1))
        self.kldiv = tf.reduce_mean(0.5 * tf.reduce_sum(
            tf.square(self.mu) + tf.square(self.sigma) -
            tf.log(1e-8 + tf.square(self.sigma)) - 1, 1))
        self.total_loss = self.kldiv + self.likelihood

    def train_step(self, x):
        _, total_loss, likelihood, kldiv = \
            self.sess.run(
                [self.train, self.total_loss, self.likelihood, self.kldiv],
                feed_dict={self.x: x})
        return total_loss, likelihood, kldiv

    def train_model(self, data, num_epochs=50, batch_size=100):
        train_loss = []
        train_lhd_loss = []
        train_kld_loss = []
        if self.saved_path is None:
            num_sample = data.num_examples
            for epoch in range(num_epochs):
                for iter in range(num_sample // batch_size):
                    batch = data.next_batch(batch_size)[0]
                    losses = self.train_step(batch)
                if epoch % 5 == 0:
                    print('[Epoch {}] Loss(total_loss, likelihood, kldiv): {}'.format(epoch, losses))
                    #print(''.format(losses[3], losses[4]))
                train_loss.append(losses[0])
                train_lhd_loss.append(losses[1])
                train_kld_loss.append(losses[2])
                if epoch % 50 == 0:
                    inter_med_path = self.model_dir+'class{}_model_hd{}_ld{}_e{}'.format('_all', self.hidden_dim, self.latent_dim, epoch)
                    self.saved_path = self.saver.save(
                                    self.sess, inter_med_path)
            # Save trained model
            self.saved_path = self.saver.save(
                self.sess, self.saving_path)
            print("Model saved in file: %s" % self.saved_path)
            y1 = np.array(train_loss)
            y2 = np.array(train_lhd_loss)
            y3 = np.array(train_kld_loss)
            t = np.arange(num_epochs)
            np.savetxt(self.saved_path+'_loss.txt', np.transpose([t, y1, y2, y3]), fmt='%-3.4f', delimiter=' ')
    # From an input x, get the reconstructed image.
    def get_reconstruct(self, x):
        reconstructed, tot_loss, likelyHd, kld = self.sess.run(
            [self.reconstruct, self.total_loss, self.likelihood, self.kldiv], feed_dict={self.x: x}
        )
        return reconstructed, tot_loss, likelyHd, kld

    def get_encoded(self, x):
        encoded = self.sess.run(self.latent, feed_dict={self.x:x})
        return encoded

    # Generate N images by sampling latent variables from normal distribution.
    def get_generated(self, N):
        random_latent = \
            np.random.normal(loc=0, scale=1, size=[N, self.latent_dim])
        return self.sess.run(
            self.reconstruct,
            feed_dict={self.latent:random_latent})
