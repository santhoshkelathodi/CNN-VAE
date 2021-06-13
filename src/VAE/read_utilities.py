# HAR classification 
# Author: Burak Himmetoglu
# 8/15/2017

import pandas as pd 
import numpy as np
import os
import cv2
import glob

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
from tensorflow.python.platform import gfile
from sklearn.model_selection import train_test_split

def read_data(data_path, split = "train"):
    """ Read data """

    # Fixed params
    n_class = 6
    n_steps = 128

    # Paths
    path_ = os.path.join(data_path, split)
    path_signals = os.path.join(path_, "Inertial_Signals")

    # Read labels and one-hot encode
    label_path = os.path.join(path_, "y_" + split + ".txt")
    labels = pd.read_csv(label_path, header = None)

    # Read time-series data
    channel_files = os.listdir(path_signals)
    channel_files.sort()
    n_channels = len(channel_files)
    posix = len(split) + 5

    # Initiate array
    list_of_channels = []
    X = np.zeros((len(labels), n_steps, n_channels))
    i_ch = 0
    for fil_ch in channel_files:
        channel_name = fil_ch[:-posix]
        dat_ = pd.read_csv(os.path.join(path_signals,fil_ch), delim_whitespace = True, header = None)
        X[:,:,i_ch] = dat_.as_matrix()

        # Record names
        list_of_channels.append(channel_name)

        # iterate
        i_ch += 1

    # Return
    return X, labels[0].values, list_of_channels

def standardize(train, test):
    """ Standardize data """

    # Standardize train and test
    X_train = (train - np.mean(train, axis=0)[None,:,:]) / np.std(train, axis=0)[None,:,:]
    X_test = (test - np.mean(test, axis=0)[None,:,:]) / np.std(test, axis=0)[None,:,:]

    return X_train, X_test

def one_hot_cust(labels, n_class = 6):
    """ One-hot encoding """
    expansion = np.eye(n_class)
    y = expansion[:, labels-1].T
    assert y.shape[1] == n_class, "Wrong number of labels!"

    return y

def get_batches(X, y, batch_size = 100):
    """ Return a generator for batches """
    n_batches = len(X) // batch_size
    X, y = X[:n_batches*batch_size], y[:n_batches*batch_size]

    # Loop over batches and yield
    for b in range(0, len(X), batch_size):
        yield X[b:b+batch_size], y[b:b+batch_size]

def get_batches(X, y, batch_size = 100):
    """ Return a generator for batches """
    n_batches = len(X) // batch_size
    X, y = X[:n_batches*batch_size], y[:n_batches*batch_size]

    # Loop over batches and yield
    for b in range(0, len(X), batch_size):
        yield X[b:b+batch_size], y[b:b+batch_size]

def get_next_batch(X, batch_size = 100):
    """ Return a generator for batches """
    n_batches = len(X) // batch_size
    X = X[:n_batches*batch_size]

    # Loop over batches and yield
    for b in range(0, len(X), batch_size):
        yield X[b:b+batch_size]

def load_train(train_path, image_width, image_height, classes):
    images = []
    labels_one_hot = []
    labels_list = []
    ids = []
    cls = []

    print('Reading training images')
    for fld in classes:   # assuming data directory has a separate folder for each class, and that each folder is named after the class
        index = classes.index(fld)
        print('Loading {} files (Index: {})'.format(fld, index))
        path = os.path.join(train_path, fld, '*g')
        files = glob.glob(path)
        for fl in files:
            image = cv2.imread(fl)
            image = cv2.resize(image, (image_width, image_height), cv2.INTER_LINEAR)
            #Start: New code to convert to hsv
            #hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            #h, s, v = cv2.split(hsv_image)
            images.append(image)
            #images.append(h)
            #End:
            labels_list.append(index+1)
            label = np.zeros(len(classes))
            label[index] = 1.0
            labels_one_hot.append(label)
            flbase = os.path.basename(fl)
            ids.append(flbase)
            cls.append(fld)
    images = np.array(images)
    labels = np.array(labels_one_hot)
    ids = np.array(ids)
    cls = np.array(cls)
    labels_list = np.array(labels_list)

    return images, labels, ids, cls, labels_list


class DataSet(object):

  def __init__(self,
               images,
               labels,
               fake_data=False,
               one_hot=False,
               dtype=dtypes.float32,
               reshape=True,
               seed=None):
    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.  Seed arg provides for convenient deterministic testing.
    """
    seed1, seed2 = random_seed.get_seed(seed)
    # If op level seed is not set, use whatever graph level seed is returned
    np.random.seed(seed1 if seed is None else seed2)
    dtype = dtypes.as_dtype(dtype).base_dtype
    if dtype not in (dtypes.uint8, dtypes.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)
    if fake_data:
      self._num_examples = 10000
      self.one_hot = one_hot
    else:
      assert images.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
      self._num_examples = images.shape[0]

      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
      if reshape:
        #print images.shape[0]
        #print images.shape[1]
        #print images.shape[2]
        #print images.shape[3]
        #assert images.shape[3] == 1
        #Start: New code for only hsv handling
        images = images.reshape(images.shape[0], images.shape[1] * images.shape[2] * images.shape[3])
        #images = images.reshape(images.shape[0], images.shape[1] * images.shape[2])
        #End
      if dtype == dtypes.float32:
        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(np.float32)
        images = np.multiply(images, 1.0 / 255.0)
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False, shuffle=True):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1] * 784
      if self.one_hot:
        fake_label = [1] + [0] * 9
      else:
        fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)
      ]
    start = self._index_in_epoch
    # Shuffle for the first epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:
      perm0 = np.arange(self._num_examples)
      np.random.shuffle(perm0)
      self._images = self.images[perm0]
      self._labels = self.labels[perm0]
    # Go to the next epoch
    if start + batch_size > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Get the rest examples in this epoch
      rest_num_examples = self._num_examples - start
      images_rest_part = self._images[start:self._num_examples]
      labels_rest_part = self._labels[start:self._num_examples]
      # Shuffle the data
      if shuffle:
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        self._images = self.images[perm]
        self._labels = self.labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch
      images_new_part = self._images[start:end]
      labels_new_part = self._labels[start:end]
      return np.concatenate((images_rest_part, images_new_part), axis=0) , np.concatenate((labels_rest_part, labels_new_part), axis=0)
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      return self._images[start:end], self._labels[start:end]


def read_data_sets_images(train_dir,
                   img_size_width,
                   img_size_height,
                   classes_images,
                   fake_data=False,
                   one_hot=False,
                   dtype=dtypes.float32,
                   reshape=True,
                   seed=None):
  if fake_data:

    def fake():
      return DataSet(
          [], [], fake_data=True, one_hot=one_hot, dtype=dtype, seed=seed)

    train = fake()
    validation = fake()
    test = fake()
    return base.Datasets(train=train, validation=validation, test=test)


  X_train_images, labels_train_images_onehot, ids, cls, labels_train_images = load_train(train_dir,
                                                                                img_size_width,
                                                                                img_size_height,
                                                                                classes_images)

  X_tr_images, X_vld_images, lab_tr_images, lab_vld_images = train_test_split(X_train_images, labels_train_images,
                                                                              stratify=labels_train_images,
                                                                              random_state=324)#random_state=123)
  validation_size = len(X_vld_images)
  if not 0 <= validation_size <= len(X_train_images):
    raise ValueError(
        'Validation size should be between 0 and {}. Received: {}.'
        .format(len(X_train_images), validation_size))
  #print len(X_train_images)
  #print len(X_tr_images)
  #print len(X_vld_images)
  validation_images = X_train_images[:validation_size]
  validation_labels = labels_train_images_onehot[:validation_size]
  train_images = X_train_images[validation_size:]
  train_labels = labels_train_images_onehot[validation_size:]


  options = dict(dtype=dtype, reshape=reshape, seed=seed)
  '''
  train = DataSet(train_images, train_labels, **options)
  validation = DataSet(validation_images, validation_labels, **options)
  test = DataSet(validation_images, validation_labels, **options)
  '''
  numClass = len(classes_images)
  #print numClass

  lab_tr_images_one_hot = one_hot_cust(lab_tr_images, numClass)
  lab_vld_images_one_hot = one_hot_cust(lab_vld_images, numClass)
  #print train_images.shape
  #print train_labels.shape
  #print X_tr_images.shape
  #print lab_tr_images_one_hot.shape
  train = DataSet(X_tr_images, lab_tr_images_one_hot, **options)
  validation = DataSet(X_vld_images, lab_vld_images_one_hot, **options)
  test = DataSet(X_vld_images, lab_vld_images_one_hot, **options)

  return base.Datasets(train=train, validation=validation, test=test)

