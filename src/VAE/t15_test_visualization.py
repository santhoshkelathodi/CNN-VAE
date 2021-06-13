import numpy as np
import matplotlib.pyplot as plt
from t15_vae import VariationalAutoEncoder
from test_utilities_autoencoder_oneshot import *
import argparse
from sklearn.manifold import TSNE

WIDTH = HEIGHT = 120
train_path = '../T15/'
test_path = '../T15/test_non_ano'
test_ano_path = '../T15/test_ano'
result_dir = '../T15/result/anomaly/single_auto_all'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
classes = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15']
#classes = ['1','2']
#classes_images = ['1']

parser = argparse.ArgumentParser(description='Parse arguments for T15 VAE.')
parser.add_argument('--reconstruct', type=int, default=1, help='reconstruction')
parser.add_argument('--generate', help='generation')
parser.add_argument('--num_gen', type=int, default=9, help='Number of examples to generate/reconstruct')
parser.add_argument('--batch_size', type=int, default=20, help='Give batch size for training')
parser.add_argument('--latent_dim', type=int, default=6, help='Give latent dimension for model')
parser.add_argument('--hidden_dim', type=int, default=500, help='Give hidden dimension for NN')
parser.add_argument('--num_epochs', type=int, default=500, help='Number of epochs to train on')
args = parser.parse_args()


img_size_height = 120#28
img_size_width = 120#28
no_channels = 3
my_image_dim = img_size_width * img_size_height * no_channels



def generate_visualisation(model, mytraindata, mytestdata, mytestanodata):
    batch_train, labels_train = mytraindata.test.next_batch(mytraindata.test.num_examples)
    batch_test, labels_test = mytestdata.test.next_batch(mytestdata.test.num_examples)
    batch_test_ano, labels_test_ano = mytestanodata.test.next_batch(mytestanodata.test.num_examples)

    latent_train = model.get_encoded(batch_train)
    latent_test = model.get_encoded(batch_test)
    latent_test_ano = model.get_encoded(batch_test_ano)
    latent_summary = np.concatenate((latent_train, latent_test), axis = 0)
    latent_summary = np.concatenate([latent_summary, latent_test_ano])
    label_summary = np.concatenate([labels_train, labels_test])
    label_summary = np.concatenate([latent_summary, labels_test_ano])
    len_train =  len(latent_train)
    len_test = len(latent_test)
    len_test_ano = len(latent_test_ano)

    #np.savetxt('../T15/result/anomaly/latent_test.out', latent, fmt='%1.4e')  # use exponential notation
    latent_embedded = TSNE(n_components=2).fit_transform(latent_summary)
    np.savetxt('../T15/result/anomaly/latent_embedded_data.out', latent_embedded, fmt='%-4.2f')
    np.savetxt('../T15/result/anomaly/label_embedded_data.out', label_summary, fmt='%-1d')
    # use exponential notation
    latent_embedded_train = TSNE(n_components=2).fit_transform(latent_train)
    latent_embedded_test = TSNE(n_components=2).fit_transform(latent_test)
    latent_embedded_test_ano = TSNE(n_components=2).fit_transform(latent_test_ano)
    print latent_train.shape
    colorList_train = np.argmax(labels_train, 1)
    colorList_test = np.argmax(labels_test, 1)
    colorList_test_ano = np.argmax(labels_test_ano, 1)
    N = 15 # number of unique elements
    print colorList_train
    fig1 = plt.figure(figsize=(8, 6))
    plt.gca().set_aspect('equal')
    x1 = latent_embedded[:len_train, 0]
    y1 = latent_embedded[:len_train, 1]
    plt.scatter(
        x1,
        y1,
        c=colorList_train,
        marker='o', edgecolor='k')
    '''
    for i, txt in enumerate(colorList):
        plt.annotate(txt, (x1[i], y1[i]))
    '''

    plt.colorbar(ticks=range(N))
    #plt.colorbar()
    plt.grid(True)
    fig1.savefig('../T15/result/anomaly/only_train_latent_space_hd{}_ld{}_e{}.png'.format(
        args.hidden_dim, args.latent_dim, args.num_epochs))
    #test
    fig2 = plt.figure(figsize=(8, 6))
    x2 = latent_embedded[len_train:(len_train+len_test), 0]
    y2 = latent_embedded[len_train:(len_train+len_test), 1]
    plt.scatter(
        x2,
        y2,
        c=colorList_test,
        marker='s', edgecolor='k')

    plt.colorbar(ticks=range(N))
    #plt.colorbar()
    plt.grid(True)
    fig2.savefig('../T15/result/anomaly/only_test_latent_space_hd{}_ld{}_e{}.png'.format(
        args.hidden_dim, args.latent_dim, args.num_epochs))
    #test_ano
    fig3 = plt.figure(figsize=(8, 6))
    x3 = latent_embedded[(len_train+len_test):, 0]
    y3 = latent_embedded[(len_train+len_test):, 1]
    plt.scatter(
        x3,
        y3,
        c=colorList_test_ano,
        marker='^', edgecolor='k')

    plt.colorbar(ticks=range(N))
    #plt.colorbar()
    plt.grid(True)
    fig3.savefig('../T15/result/anomaly/test_ano_latent_space_hd{}_ld{}_e{}.png'.format(
        args.hidden_dim, args.latent_dim, args.num_epochs))

from mpl_toolkits.mplot3d import Axes3D

def generate_3dvisualisation(model, mytraindata, mytestdata, mytestanodata):
    batch_train, labels_train = mytraindata.test.next_batch(mytraindata.test.num_examples)
    batch_test, labels_test = mytestdata.test.next_batch(mytestdata.test.num_examples)
    batch_test_ano, labels_test_ano = mytestanodata.test.next_batch(mytestanodata.test.num_examples)

    latent_train = model.get_encoded(batch_train)
    latent_test = model.get_encoded(batch_test)
    latent_test_ano = model.get_encoded(batch_test_ano)
    latent_summary = np.concatenate((latent_train, latent_test), axis = 0)
    latent_summary = np.concatenate([latent_summary, latent_test_ano])
    len_train =  len(latent_train)
    len_test = len(latent_test)
    len_test_ano = len(latent_test_ano)

    #np.savetxt('../T15/result/anomaly/latent_test.out', latent_summary, fmt='%1.2f')  # use exponential notation
    latent_embedded = TSNE(n_components=2).fit_transform(latent_summary)
    #np.savetxt('../T15/result/anomaly/latent_data.out', latent_summary, fmt='%1.2f')  # use exponential notation

    print latent_train.shape
    colorList_train = np.argmax(labels_train, 1)
    colorList_test = np.argmax(labels_test, 1)
    colorList_test_ano = np.argmax(labels_test_ano, 1)
    N = 15 # number of unique elements
    print colorList_train
    #fig1 = plt.figure(figsize=(8, 6))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    z1 = np.full((1, len_train), 10)
    x1 = latent_embedded[:len_train, 0]
    y1 = latent_embedded[:len_train, 1]
    ax.scatter(
        x1,
        y1,
        z1,
        c=colorList_train,
        marker='o', edgecolor='k')
    '''
    for i, txt in enumerate(colorList):
        plt.annotate(txt, (x1[i], y1[i]))
    '''
    #fig1.savefig('../T15/result/anomaly/only_train_latent_space_hd{}_ld{}_e{}.png'.format(
    #    args.hidden_dim, args.latent_dim, args.num_epochs))
    #test
    #fig2 = plt.figure(figsize=(8, 6))
    x2 = latent_embedded[len_train:(len_train+len_test), 0]
    y2 = latent_embedded[len_train:(len_train+len_test), 1]
    z2 = np.full((1, len_test), 20)
    ax.scatter(
        x2,
        y2,
        z2,
        c=colorList_test,
        marker='s', edgecolor='k')
    #fig2.savefig('../T15/result/anomaly/only_test_latent_space_hd{}_ld{}_e{}.png'.format(
    #    args.hidden_dim, args.latent_dim, args.num_epochs))
    #test_ano
    #fig3 = plt.figure(figsize=(8, 6))
    x3 = latent_embedded[(len_train+len_test):, 0]
    y3 = latent_embedded[(len_train+len_test):, 1]
    z3 = np.full((1, len_test_ano), 30)
    ax.scatter(
        x3,
        y3,
        z3,
        c=colorList_test_ano,
        marker='^', edgecolor='k')

    #plt.colorbar(ticks=range(N))
    #plt.colorbar()
    #plt.grid(True)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()
    #fig.savefig('../T15/result/anomaly/3d_together_test_ano_latent_space_hd{}_ld{}_e{}.png'.format(
    #    args.hidden_dim, args.latent_dim, args.num_epochs))


def generate_labeling_visualisation(model, images, one_hot_labels, id, dtype = dtypes.float32, reshape=True):
    batch_train, labels_train = mytraindata.test.next_batch(mytraindata.test.num_examples)

    # to [num examples, rows*columns] (assuming depth == 1)
    if reshape:
        #Start: New code for only hsv handling
        images = images.reshape(images.shape[0], images.shape[1] * images.shape[2] * images.shape[3])
    if dtype == dtypes.float32:
        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(np.float32)
        images = np.multiply(images, 1.0 / 255.0)
    latent_batch = model.get_encoded(images) # convert to latent features
    '''
    #latent_test = model.get_encoded(batch_test)
    #latent_test_ano = model.get_encoded(batch_test_ano)
    latent_summary = np.concatenate((latent_train, latent_test), axis = 0)
    latent_summary = np.concatenate([latent_summary, latent_test_ano])
    label_summary = np.concatenate([labels_train, labels_test])
    label_summary = np.concatenate([latent_summary, labels_test_ano])
    len_train =  len(latent_train)
    len_test = len(latent_test)
    len_test_ano = len(latent_test_ano)
    '''

    #np.savetxt('../T15/result/anomaly/latent_test.out', latent, fmt='%1.4e')  # use exponential notation
    latent_embedded = TSNE(n_components=2).fit_transform(latent_batch)
    np.savetxt('../T15/result/anomaly/latent_embedded_data.out', latent_embedded, fmt='%-4.2f')
    np.savetxt('../T15/result/anomaly/name_embedded_data.out', id, fmt='%s')
    '''
    latent_embedded_train = TSNE(n_components=2).fit_transform(latent_train)
    latent_embedded_test = TSNE(n_components=2).fit_transform(latent_test)
    latent_embedded_test_ano = TSNE(n_components=2).fit_transform(latent_test_ano)
    '''
    #print latent_train.shape
    colorList_train = np.argmax(one_hot_labels, 1)
    len_train = len(latent_batch)
    np.savetxt('../T15/result/anomaly/label_embedded_data.out', colorList_train, fmt='%-1d')
    #colorList_test = np.argmax(labels_test, 1)
    #colorList_test_ano = np.argmax(labels_test_ano, 1)
    N = 15 # number of unique elements
    print colorList_train
    fig1 = plt.figure(figsize=(8, 6))
    plt.gca().set_aspect('equal')
    x1 = latent_embedded[:len_train, 0]
    y1 = latent_embedded[:len_train, 1]
    plt.scatter(
        x1,
        y1,
        c=colorList_train,
        marker='o', edgecolor='k')
    '''
    for i, txt in enumerate(colorList):
        plt.annotate(txt, (x1[i], y1[i]))
    '''

    plt.colorbar(ticks=range(N))
    #plt.colorbar()
    plt.grid(True)
    fig1.savefig('../T15/result/anomaly/only_train_latent_space_hd{}_ld{}_e{}.png'.format(
        args.hidden_dim, args.latent_dim, args.num_epochs))

import tensorflow as tf

#read the data

mytraindata, labels_train_images, nameList = read_data_sets_images(train_path,
                                                            img_size_width,
                                                            img_size_height,
                                                            classes)
'''
mytestdata, labels_test_images, testnameList = read_data_sets_images(test_path,
                                                            img_size_width,
                                                            img_size_height,
                                                            classes)

mytestanodata, labels_test_no_images, test_ano_nameList = read_data_sets_images(test_ano_path,
                                                            img_size_width,
                                                            img_size_height,
                                                            classes)
                                                            '''
# saved model
model_save_path = 'trained_model/class{}_model_hd{}_ld{}_e{}'.format('_all', 500, 6, 500)
print model_save_path

tf.reset_default_graph()
model = VariationalAutoEncoder(
    dim_image=my_image_dim,
    batch_size=args.batch_size,
    latent_dim=args.latent_dim,
    hidden_dim=args.hidden_dim,
    saved_path=model_save_path,
    test_flag=1)

X_train_images, labels_train_images_onehot, ids, cls, labels_train_images = load_train(train_path,
                                                                                img_size_width,
                                                                                img_size_height,
                                                                                       classes)

#generate_3dvisualisation(model, mytraindata, mytestdata, mytestanodata)
generate_labeling_visualisation(model, X_train_images, labels_train_images_onehot, ids)

