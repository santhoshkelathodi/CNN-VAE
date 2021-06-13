import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from t15_vae import VariationalAutoEncoder
from read_utilities import *
import argparse

WIDTH = HEIGHT = 120

## Parsing arguments
parser = argparse.ArgumentParser(description='Parse arguments for T15 VAE.')
parser.add_argument('--batch_size', type=int, default=20, help='Give batch size for training')
parser.add_argument('--latent_dim', type=int, default=6, help='Give latent dimension for model')
parser.add_argument('--hidden_dim', type=int, default=500, help='Give hidden dimension for NN')
parser.add_argument('--num_epochs', type=int, default=300, help='Number of epochs to train on')
parser.add_argument('--num_gen', type=int, default=9, help='Number of examples to generate')
parser.add_argument('--saving_path', type=str, default=None, help='path to save model')
args = parser.parse_args()

train_path = './T15_REFINED_TESTING_GPU/'
model_dir = './T15_REFINED_TESTING_GPU/trained_model/'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
classes_images = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15']
#classes_images = ['1']

img_size_height = 120#28
img_size_width = 120#28
no_channels = 3
mydata = read_data_sets_images(train_path,
                               img_size_width,
                               img_size_height,
                               classes_images)

my_image_dim = mydata.train.images[0].shape[0]
my_num_sample = mydata.train.num_examples
print(my_image_dim)
print(my_num_sample)

mylatent_dim = len(classes_images)
model_save_path = model_dir+'class{}_model_hd{}_ld{}_e{}'.format('_all', args.hidden_dim, args.latent_dim, args.num_epochs)
#model_save_path = 'trained_model/class{}_model_hd{}_ld{}_e{}'.format('_all_test_', args.hidden_dim, args.latent_dim, args.num_epochs)
print(model_save_path)

mymodel = VariationalAutoEncoder(
    lr=5e-4,
    dim_image=my_image_dim,
    batch_size=args.batch_size,
    latent_dim=args.latent_dim,
    hidden_dim=args.hidden_dim,
    num_epochs=args.num_epochs,
    saving_path = model_save_path,
    model_dir = model_dir)

mymodel.train_model(mydata.train, num_epochs=args.num_epochs, batch_size=args.batch_size)
