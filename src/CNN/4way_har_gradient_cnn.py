# Imports
import numpy as np
import os
from sklearn.metrics import confusion_matrix
from utilities import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#matplotlib inline

#newcode
train_path = 'RUPALI/'
test_path = 'RUPALI/' # as of now giving both training and test as same data. But need to segregate later
checkpoint_dir = "RUPALI/models/"
results_dir = "RUPALI/results/"
result_prefix = "rupali_"
img_size_height = 120
img_size_width = 120

#classes_images = ['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6', 'Class 7', 'Class 8', 'Class 9', 'Class 10', 'Class 11', 'Class 12', 'Class 13', 'Class 14', 'Class 15', 'Class 16', 'Class 17', 'Class 18', 'Class 19', 'Class 20']
#classes_images = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20']
classes_images = ['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6', 'Class 8', 'Class 9', 'Class 10', 'Class 11', 'Class 12', 'Class 13', 'Class 14', 'Class 15', 'Class 16', 'Class 18', 'Class 19', 'Class 20']
n_class_images = 18 #RUPALI
X_train_images, labels_train_images_onehot, ids, cls, labels_train_images = load_train(train_path, img_size_width, img_size_height, classes_images)
#X_test_images, labels_test_images_onehot, ids, cls, labels_test_images = load_train(test_path, img_size_width, img_size_height, classes_images)
#newcode

print

#newcode
#X_tr_images, X_vld_images, lab_tr_images, lab_vld_images = train_test_split(X_train_images, labels_train_images,
#                                                stratify = labels_train_images, random_state = 123)
X_tr_images, X_vld_images, lab_tr_images, lab_vld_images = train_test_split(X_train_images, labels_train_images,
                                                stratify = labels_train_images, test_size=0.25, random_state = 42)

#newcode

#newcode
y_tr_images = one_hot(lab_tr_images, n_class_images)
y_vld_images = one_hot(lab_vld_images, n_class_images)
#y_test_images = one_hot(labels_test_images, n_class_images)
print np.unique(lab_vld_images)
print lab_vld_images[0]
print y_vld_images[0]
# Imports
import tensorflow as tf
#newcode
batch_size_images = 20 # 200 good #20 originally tested 600 dont test. slows down your machine      # Batch size
seq_len_images =  img_size_width * img_size_height         # Number of steps
learning_rate_images = 0.0001
epochs_images = 10 #50
n_channels_images = 3	

graph_images = tf.Graph()
#newcode

#newcode
#http://cv-tricks.com/tensorflow-tutorial/training-convolutional-neural-network-for-image-classification/
# Construct placeholders
with graph_images.as_default():
    #doubt whether to reshape here?
    images_inputs_ = tf.placeholder(tf.float32, [None, img_size_width, img_size_height, n_channels_images], name='inputs')
    images_labels_ = tf.placeholder(tf.float32, [None, n_class_images], name='labels')
    images_keep_prob_ = tf.placeholder(tf.float32, name='keep')
    images_learning_rate_ = tf.placeholder(tf.float32, name='learning_rate')
#newcode

#newcode
with graph_images.as_default():
    # (batch, 128, 9) --> (batch, 64, 18)
    images_conv1 = tf.layers.conv2d(inputs=images_inputs_, filters=18, kernel_size=2, strides=1,
                             padding='same', activation=tf.nn.relu)
    images_max_pool_1 = tf.layers.max_pooling2d(inputs=images_conv1, pool_size=2, strides=2, padding='same')

    # (batch, 64, 18) --> (batch, 32, 36)
    images_conv2 = tf.layers.conv2d(inputs=images_max_pool_1, filters=36, kernel_size=2, strides=1,
                             padding='same', activation=tf.nn.relu)
    images_max_pool_2 = tf.layers.max_pooling2d(inputs=images_conv2, pool_size=2, strides=2, padding='same')

    # (batch, 32, 36) --> (batch, 16, 72)
    images_conv3 = tf.layers.conv2d(inputs=images_max_pool_2, filters=72, kernel_size=2, strides=1,
                             padding='same', activation=tf.nn.relu)
    images_max_pool_3 = tf.layers.max_pooling2d(inputs=images_conv3, pool_size=2, strides=2, padding='same')

    # (batch, 16, 72) --> (batch, 8, 144)
    images_conv4 = tf.layers.conv2d(inputs=images_max_pool_3, filters=144, kernel_size=2, strides=1,
                             padding='same', activation=tf.nn.relu)
    images_max_pool_4 = tf.layers.max_pooling2d(inputs=images_conv4, pool_size=2, strides=2, padding='same')
#newcode


#newcode
with graph_images.as_default():
    # Flatten and add dropout
    images_flat = tf.reshape(images_max_pool_4, (-1, 8 * 8 * 144))#for 2-D for 1-D 8*144 was enough.Bug fixed
    images_flat = tf.nn.dropout(images_flat, keep_prob=images_keep_prob_)

    # Predictions
    images_logits = tf.layers.dense(images_flat, n_class_images)
    images_logits_raw = tf.identity(images_logits, name='images_logits')
    pred_class = tf.argmax(images_logits, 1, name = 'pred_class')
    input_class = tf.argmax(images_labels_, 1, name = 'input_class')
    y_pred = tf.nn.softmax(images_logits,name='y_pred')

    # Cost function and optimizer
    images_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=images_logits, labels=images_labels_), name='cost')
    images_optimizer = tf.train.AdamOptimizer(images_learning_rate_).minimize(images_cost)

    # Accuracy
    #images_correct_pred = tf.equal(tf.argmax(images_logits, 1), tf.argmax(images_labels_, 1))
    images_correct_pred = tf.equal(pred_class, input_class)
    images_accuracy = tf.reduce_mean(tf.cast(images_correct_pred, tf.float32), name='accuracy')
#newcode

#newcode
images_validation_acc = []
images_validation_loss = []

images_train_acc = []
images_train_loss = []
#newcode


#newcode
with graph_images.as_default():
    images_saver = tf.train.Saver()
#newcode
'''
#newcode
with tf.Session(graph=graph_images) as sess_images:
    sess_images.run(tf.global_variables_initializer())
    images_iteration = 1

    # Loop over epochs
    for e in range(epochs_images):

        # Loop over batches
        for x, y in get_batches(X_tr_images, y_tr_images, batch_size_images):

            # Feed dictionary
            feed = {images_inputs_: x, images_labels_: y, images_keep_prob_: 0.5, images_learning_rate_: learning_rate_images}

            # Loss
            loss, _, acc = sess_images.run([images_cost, images_optimizer, images_accuracy], feed_dict=feed)
            images_train_acc.append(acc)
            images_train_loss.append(loss)

            # Print at each 5 iters
            if (images_iteration % 5 == 0):
                print("Epoch: {}/{}".format(e, epochs_images),
                      "Iteration: {:d}".format(images_iteration),
                      "Train loss: {:6f}".format(loss),
                      "Train acc: {:.6f}".format(acc))

            # Compute validation loss at every 10 iterations
            if (images_iteration % 10 == 0):
                val_acc_ = []
                val_loss_ = []

                for x_v, y_v in get_batches(X_vld_images, y_vld_images, batch_size_images):
                    # Feed
                    feed = {images_inputs_: x_v, images_labels_: y_v, images_keep_prob_: 1.0}

                    # Loss
                    loss_v, acc_v = sess_images.run([images_cost, images_accuracy], feed_dict=feed)
                    val_acc_.append(acc_v)
                    val_loss_.append(loss_v)

                # Print info
                print("Epoch: {}/{}".format(e, epochs_images),
                      "Iteration: {:d}".format(images_iteration),
                      "Validation loss: {:6f}".format(np.mean(val_loss_)),
                      "Validation acc: {:.6f}".format(np.mean(val_acc_)))

                # Store
                images_validation_acc.append(np.mean(val_acc_))
                images_validation_loss.append(np.mean(val_loss_))

            # Iterate
            images_iteration += 1
        saveName =  checkpoint_dir+"all_class_overall.ckpt"
        #print saveName
        images_saver.save(sess_images, saveName)
#newcode

#newcode
# Plot training and test loss
#images_iteration = images_iteration/10*10
t_images = np.arange(images_iteration-1)
t_val_images = t_images[t_images % 10 == 0]
#print len(t_val_images)
#print t_images
#print t_val_images
if (len(t_val_images) > len(images_validation_loss)):
    t_val_images = t_val_images[:-1].copy()
#print t_images
#print len(t_images)
#print len(images_train_loss)
#print len(images_validation_loss)
#print len(t_val_images)
fig = plt.figure(figsize = (6,4))
plt.plot(t_images, np.array(images_train_loss), 'r-', t_val_images, np.array(images_validation_loss), 'b*')
#plt.plot(t_images, np.array(images_train_loss), 'r-')
#plt.plot(t_images[t_images % 10 == 0], np.array(images_validation_loss), 'b*')
plt.xlabel("iteration")
plt.ylabel("Loss")
plt.legend(['train', 'validation'], loc='upper right')
#plt.show()

fig.savefig(results_dir+'t15_loss_plot.png')

# Plot Accuracies
fig = plt.figure(figsize = (6,4))

plt.plot(t_images, np.array(images_train_acc), 'r-', t_val_images, images_validation_acc, 'b*')
plt.xlabel("iteration")
plt.ylabel("Accuray")
plt.legend(['train', 'validation'], loc='upper right')
#plt.show()
fig.savefig(results_dir+'t15_laccuracy_plot.png')
#newcode
#'''

# Helper-function to plot confusion matrix
def plot_confusion_matrix(cls_pred,cls_true, num_classes):
    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.
    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)

    # Print the confusion matrix as text.
    print(cm)
    # Plot the confusion matrix as an image.
    plt.matshow(cm)
    # Make various adjustments to the plot.
    #plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    #plt.show()
    plt.savefig(results_dir+'t15_confusion_mat.png', cmap='grey')

'''
Parameters:
    correct_labels                  : These are your true classification categories.
    predict_labels                  : These are you predicted classification categories
    labels                          : This is a lit of labels which will be used to display the axix labels
    title='Confusion matrix'        : Title for your matrix
    tensor_name = 'MyFigure/image'  : Name for the output summay tensor

Returns:
    summary: TensorFlow summary

Other itema to note:
    - Depending on the number of category and the data , you may have to modify the figzie, font sizes etc.
    - Currently, some of the ticks dont line up due to rotations.
'''
from textwrap import wrap
import re
import itertools
import matplotlib
def plot_confusion_matrix_cust(correct_labels,
                               predict_labels,
                               labels,
                               title='Confusion matrix',
                               tensor_name = 'MyFigure/image',
                               normalize=False):
    cm = confusion_matrix(correct_labels, predict_labels, labels=labels)
    if normalize:
        cm = cm.astype('float')*10 / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm, copy=True)
        cm = cm.astype('int')

    np.set_printoptions(precision=2)
    ###fig, ax = matplotlib.figure.Figure()

    fig = matplotlib.figure.Figure(figsize=(7, 7), dpi=320, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(cm, cmap='Oranges')

    classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in labels]
    classes = ['\n'.join(wrap(l, 40)) for l in classes]

    tick_marks = np.arange(len(classes))

    ax.set_xlabel('Predicted', fontsize=7)
    ax.set_xticks(tick_marks)
    c = ax.set_xticklabels(classes, fontsize=4, rotation=-90,  ha='center')
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    ax.set_ylabel('True Label', fontsize=7)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=4, va ='center')
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], 'd') if cm[i,j]!=0 else '.', horizontalalignment="center", fontsize=6, verticalalignment='center', color= "black")
    fig.set_tight_layout(True)
    #summary = tfplot.figure.to_summary(fig, tag=tensor_name)
    return #summary
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
        value = int(round(cm[int(y_val)][int(x_val)]))
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
# End:Plot grey scale confusion matrix
test_acc_images = []
with tf.Session(graph=graph_images) as sess_images:
    # Restore
    images_saver.restore(sess_images, tf.train.latest_checkpoint(checkpoint_dir))

    #start:new code
    num_test_images = len(X_vld_images)
    print num_test_images
    print y_vld_images[0]
    #x_t, y_t = get_batches(X_vld_images, y_vld_images, num_test_images)
    feed = {images_inputs_: X_vld_images,
            images_labels_: y_vld_images,
            images_keep_prob_: 1}
    #batch_acc_images = sess_images.run(images_accuracy, feed_dict=feed)
    result, input = sess_images.run([pred_class,input_class], feed_dict=feed)

    #print res_pred1[0]
    #print result_pred
    #print lab_vld_images
    plot_confusion_matrix(result, input, n_class_images)
    cm = confusion_matrix(y_true=result,
                          y_pred=input)
    print(cm)
    '''
    # Transform to percentage
    for row in range(0, len(cm)):
        rowSum = np.sum(cm[row])
        cm[row] = cm[row] / float(rowSum) * 100

    '''
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm)
    cm = cm*100
    print(cm)

    data_labels = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18']
    plotMatrix(cm, data_labels, "IITBBSR", results_dir+result_prefix+'grey_confusion_mat.png')

    #end:new code


    #for x_t, y_t in get_batches(X_test_images, y_test_images, batch_size_images):
    #for x_t, y_t in get_batches(X_vld_images, y_vld_images, batch_size_images):
    #start: old code
    for x_t, y_t in get_batches(X_vld_images, y_vld_images, batch_size_images):
        feed = {images_inputs_: x_t,
                images_labels_: y_t,
                images_keep_prob_: 1}

        batch_acc_images = sess_images.run(images_accuracy, feed_dict=feed)
        test_acc_images.append(batch_acc_images)
    print("Test accuracy: {:.6f}".format(np.mean(test_acc_images)))

    #end: oldcode
    # '''
