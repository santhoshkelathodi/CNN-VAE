import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from t15_vae import VariationalAutoEncoder
from test_utilities_autoencoder_oneshot import *
import argparse
from sklearn.manifold import TSNE

WIDTH = HEIGHT = 120
test_path = './T15_REFINED_TESTING_GPU/anomaly/single_auto_all/'
result_dir = './T15_REFINED_TESTING_GPU/result/anomaly/single_auto_all'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
model_dir = './T15_REFINED_TESTING_GPU/trained_model/' #for refined trajectories
#model_dir = 'trained_model/' # nonrefined model
#classes = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15']
#classes = ['1','2']
#classes_images = ['1']
multfactor = 2.0
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

def test_anomaly_1(model, class_index, mydata):

    print("testing model", class_index)
    num_test_images = mydata.test.num_examples
    print(num_test_images)
    in_imgs = mydata.test.images[:num_test_images]
    images = []
    predlabel = []
    for image in in_imgs:
        batch,label = mydata.test.next_batch(1)
        images.append(image)
        batch1 = np.array(images)
        x_reconstructed, tot_loss, likelyhd, kld = model.get_reconstruct(batch1)
        print(tot_loss)
        if (tot_loss > (t15_threshold[class_index]*multfactor)):
            predlabel.append(1)
        else:
            predlabel.append(2)
        images = []
    # saving figure for cross reference
    fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(20,4))
    x_reconstructed, tot_loss, likelyhd, kld = model.get_reconstruct(in_imgs)
    for images, row in zip([in_imgs, x_reconstructed], axes):
        for img, ax in zip(images, row):
            ax.imshow(img.reshape((img_size_width, img_size_height, no_channels)))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    fig.tight_layout(pad=0.1)
    fig.savefig(result_dir+'oneshot_I_reconstructed_anomaly_class{}_hd{}_ld{}_e{}.png'.format(class_index+1,
        args.hidden_dim, args.latent_dim, args.num_epochs))
    #end of figure saving

    return np.array(predlabel)


def test_anomaly_2(model, threshold, mydata):

    print("testing threshold", threshold)
    num_test_images = mydata.test.num_examples
    print(num_test_images)
    in_imgs = mydata.test.images[:num_test_images]
    images = []
    predlabel = []
    lossList = []
    lossLikelihood = []
    lossKLD = []
    for image in in_imgs:
        batch,label = mydata.test.next_batch(1)
        images.append(image)
        batch1 = np.array(images)
        x_reconstructed, tot_loss, likelyhd, kld = model.get_reconstruct(batch1)
        print(tot_loss)
        lossList.append(tot_loss)
        lossLikelihood.append(likelyhd)
        lossKLD.append(kld)
        if (tot_loss > (threshold*multfactor)):
            predlabel.append(1)
        else:
            predlabel.append(2)
        images = []
    # saving figure for cross reference
    fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(20,4))
    x_reconstructed, tot_loss, likelyhd, kld = model.get_reconstruct(in_imgs)
    for images, row in zip([in_imgs, x_reconstructed], axes):
        for img, ax in zip(images, row):
            ax.imshow(img.reshape((img_size_width, img_size_height, no_channels)))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    fig.tight_layout(pad=0.1)
    fig.savefig(result_dir+'oneshot_I_reconstructed_anomaly_class_all_hd{}_ld{}_e{}.png'.format(
                                                                args.hidden_dim,
                                                                args.latent_dim,
                                                                args.num_epochs))
    #end of figure saving

    return np.array(predlabel), np.array(lossList), np.array(lossLikelihood), np.array(lossKLD)

# Start:Plot grey scale confusion matrix
LABEL_SIZE = 14
VALUE_SIZE = 12
LABEL_ROTATION = 0
import matplotlib.colors as colors
import matplotlib.patches as patches
# Plot the matrix: show and save as image
def plotMatrix(cm, labels, title, fname):
    print("> Plot confusion matrix to", fname, "...")
    fig = plt.figure()

    # COLORS
    # ======
    # Discrete color map: make a color map of fixed colors
    # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    color_list = []
    for i in range(10):
        i += 1
        color_list.append((i*0.1, i*0.1, i*0.1))
    # Reversed gray scale (black for 100, white for 0)
    color_map = colors.ListedColormap(list(reversed(color_list)))
    # Set color bounds: [0,10,20,30,40,50,60,70,80,90,100]
    bounds=range(0, 110, 10)
    norm = colors.BoundaryNorm(bounds, color_map.N)

    # Plot matrix (convert numpy to list) and set Z max to 100
    plt.imshow(cm, interpolation='nearest', cmap=color_map, vmax=100)
    plt.title(title)
    plt.colorbar()

    # LABELS
    # ======
    # Setup labels (same for both axises)
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, fontsize=LABEL_SIZE, rotation=LABEL_ROTATION)
    plt.yticks(tick_marks, labels, fontsize=LABEL_SIZE)
    plt.ylabel('True labels')
    plt.xlabel('Predicted labels')

    # VALUES
    # ======
    # Add value text on the plot
    ax = fig.add_subplot(1, 1, 1)
    min_val, max_val, diff = 0., len(labels), 1.
    ind_array = np.arange(min_val, max_val, diff)
    x, y = np.meshgrid(ind_array, ind_array)
    # Display values on the correct position
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        # Round the float numbers
        value = round(cm[int(y_val)][int(x_val)],1)
        # Only show values that are not 0
        if value != 0:
            # Draw boxes
            ax.add_patch(
                patches.Rectangle(
                    (x_val-0.5, y_val-0.5),   # (x,y)
                    1,          # width
                    1,          # height
                    fill=None,
                    edgecolor=(0.8, 0.8, 0.8),
                )
            )
            # Show lighter color for dark background
            if value > 50:
                ax.text(x_val, y_val, value, va='center', ha='center', fontsize=VALUE_SIZE, color=(1, 1, 1))
            elif value >= 10:
                ax.text(x_val, y_val, value, va='center', ha='center', fontsize=VALUE_SIZE, color=(0, 0, 0))
            else:
                ax.text(x_val, y_val, value, va='center', ha='center', fontsize=VALUE_SIZE, color=(0, 0, 0))

    # Hide the little ticks on the axis
    for tic in ax.xaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False
    for tic in ax.yaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False

    # Save as an image
    # Higher DPI to avoid mis-alignment
    plt.savefig(fname, dpi=240)
    # Show the plot in a window
    plt.show()


import tensorflow as tf

classes_images = ['yes','no']
#classes_images = ['no']
predList = []
trueList = []
classes = ['single_auto_all']
for fld in classes:  # assuming data directory has a separate folder for each class, and that each folder is named after the class
    print(fld)
    # saved model
    model_save_path = model_dir+'class{}_model_hd{}_ld{}_e{}'.format('_all', 500, 6, 500)
    print(model_save_path)
    index = classes.index(fld)

    src_path = test_path
    mydata, labels_train_images, nameList = read_data_sets_images(src_path,
                                    img_size_width,
                                    img_size_height,
                                    classes_images)

    tf.reset_default_graph()
    model = VariationalAutoEncoder(
        dim_image=my_image_dim,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        saved_path=model_save_path,
        test_flag=1)
    trueList = np.concatenate([trueList, labels_train_images])
    t15_threshold = 600.0
    predClassList, lossArray, likelyArray, kldArray = test_anomaly_2(model, t15_threshold, mydata)
    predList = np.concatenate([predList, predClassList])

np.savetxt(result_dir+'testTrue.out', np.c_[trueList, predList, lossArray, likelyArray, kldArray, nameList], fmt="%s")
print(trueList)
print(predList)
np.savetxt(result_dir+'all_anomalyresult.out', (trueList,predList), fmt = '%d')
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true=predList,
                      y_pred=trueList)
print(cm)
# Transform to percentage

cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print(cm)
cm = cm * 100
print(cm)

data_labels = ['Anomalous', 'Normal']
plotMatrix(cm, data_labels, "T15-Anomaly-Confusion Matrix", result_dir+'2_5_grey_confusion_mat.png')
