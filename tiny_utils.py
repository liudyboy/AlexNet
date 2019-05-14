import numpy as np
import random
import chainer
from chainer.dataset import dataset_mixin
import numpy
import six
import os
from PIL import Image

class labeledImageDataSet(dataset_mixin.DatasetMixin):
    def __init__(self, pairs, root='.', dtype=None, label_dtype=numpy.int32):
        if isinstance(pairs, six.string_types):
            pairs_path = pairs
            with open(pairs_path) as pairs_file:
                pairs = []
                for i, line in enumerate(pairs_file):
                    pair = line.strip().split()
                    if len(pair) != 2:
                        raise ValueError(
                            'invalid format at line {} in file {}'.format(
                                i, pairs_path))
                    pairs.append((pair[0], int(pair[1])))
        self._pairs = pairs
        self._root = root
        self._dtype = chainer.get_dtype(dtype)
        self._label_dtype = label_dtype

    def __len__(self):
        return len(self._pairs)

    def get_batch_data(self, index):
        X = []
        Y = []
        for i in index:
            x, y = self.get_example(i)
            while(x.shape != (64, 64, 3)):
                t = np.random.randint(100000, size=1)
                x, y = self.get_example(t[0])
            X.append(x)
            Y.append(y)
        return np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.float32)

    def get_example(self, i):
        x, y = self.get_image(i)
        while(x.shape != (64, 64, 3)):
            t = np.random.randint(100000, size=1)
            x, y = self.get_image(t[0])

        imgMean = np.array([104, 117, 124], np.float)
        x = x - imgMean
        return x, y

    def get_image(self, i):
        path, int_label = self._pairs[i]
        full_path = os.path.join(self._root, path)
        image = Image.open(full_path)
        # image = image.resize([227, 227])
        # image.show()
        img = numpy.asarray(image, dtype=np.float32)
        image.close()
        label = numpy.array(int_label, dtype=self._label_dtype)
        return (img), label



def get_batch_data(batch_size):
    """read data images from tiny imageNet files """
    dataset = labeledImageDataSet('./tiny_train.txt', 'train')

    MAX_NUM = 100000    #the total number of image
    index = np.random.randint(MAX_NUM, size=batch_size) # randomly get the samples
    # index = np.arange(batch_size)
    X, Y = dataset.get_batch_data(index)
    X = np.transpose(X, (0, 3, 1, 2))
    return X, Y

def randomize(x, y):
    """ Randomizes the order of data samples and their corresponding labels"""
    permutation = np.random.permutation(y.shape[0])
    shuffled_x = x[permutation, :, :, :]
    shuffled_y = y[permutation]
    return shuffled_x, shuffled_y



def reformat(x, y):
    """
    Reformats the data to the format acceptable for convolutional layers
    :param x: input array
    :param y: corresponding labels
    :return: reshaped input and labels
    """
    img_size, num_ch, num_class = int(np.sqrt(x.shape[-1])), 1, len(np.unique(y))
    dataset = x.reshape((-1, img_size, img_size, num_ch)).astype(np.float32)
    labels = (np.arange(num_class) == y[:, None]).astype(np.float32)
    return dataset, labels


def get_next_batch(x, y, batch_size = 1, start = 0):
    if end > x.shape[0]:
        x_batch = x[start:batch_size]
        y_batch = y[start:batch_size]
    else:
        x_batch = x[start:x.shape[0]]
        y_batch = y[start:y.shape[0]]
    return x_batch, y_batch

def plot_images(images, cls_true, cls_pred=None, title=None):
    """
    Create figure with 3x3 sub-plots.
    :param images: array of images to be plotted, (9, img_h*img_w)
    :param cls_true: corresponding true labels (9,)
    :param cls_pred: corresponding true labels (9,)
    """
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(np.squeeze(images[i]), cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            ax_title = "True: {0}".format(cls_true[i])
        else:
            ax_title = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        ax.set_title(ax_title)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    if title:
        plt.suptitle(title, size=20)
    plt.show(block=False)


def plot_example_errors(images, cls_true, cls_pred, title=None):
    """
    Function for plotting examples of images that have been mis-classified
    :param images: array of all images, (#imgs, img_h*img_w)
    :param cls_true: corresponding true labels, (#imgs,)
    :param cls_pred: corresponding predicted labels, (#imgs,)
    """
    # Negate the boolean array.
    incorrect = np.logical_not(np.equal(cls_pred, cls_true))

    # Get the images from the test-set that have been
    # incorrectly classified.
    incorrect_images = images[incorrect]

    # Get the true and predicted classes for those images.
    cls_pred = cls_pred[incorrect]
    cls_true = cls_true[incorrect]

    # Plot the first 9 images.
    plot_images(images=incorrect_images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9],
                title=title)

def softmax1d(input_data):
    exps = np.exp(input_data - np.max(input_data))

    return exps / np.sum(exps)


def softmax2d(input_data):
    """
    Args:
      input_data: array like data, two dimension
    """
    input_data = np.array(input_data, dtype=np.float32)

    result = np.zeros_like(input_data)
    for i in np.arange(input_data.shape[0]):
        result[i, :] = softmax1d(input_data[i, :])
    return result


def cross_entropy2d(logits, labels):
    """
    logits is softmaxed output from fully connected layers
    labels is one hot encode vector
    """

    logits = np.asarray(logits)
    labels = np.asarray(labels, dtype=np.int)
    l = logits.shape[0]
    log_likelihood = -np.log(logits[range(l), np.argmax(labels, axis=1)])
    loss = np.sum(log_likelihood) / l
    return loss 
    

def max_pool_grad(input, out, grad_out, filter_w, filter_h):
    d_input = np.zeros_like(input)


    for i in np.arange(out.shape[0]):
        for j in np.arange(out.shape[1]):
            for k in np.arange(out.shape[2]):
                for l in np.arange(out.shape[3]):
                  pass
















