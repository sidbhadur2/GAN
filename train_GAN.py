# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Routine for decoding the CIFAR-10 binary file format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange  # pylint: disable=redefined-builtin
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

from model import discriminator, generator
import tensorflow as tf
import numpy as np
import time

# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 32

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000


def read_cifar10(filename_queue):
    """Reads and parses examples from CIFAR10 data files.

    Recommendation: if you want N-way read parallelism, call this function
    N times.  This will give you N independent Readers reading different
    files & positions within those files, which will give better mixing of
    examples.

    Args:
        filename_queue: A queue of strings with the filenames to read from.

    Returns:
        An object representing a single example, with the following fields:
            height: number of rows in the result (32)
            width: number of columns in the result (32)
            depth: number of color channels in the result (3)
            key: a scalar string Tensor describing the filename & record number
                for this example.
            label: an int32 Tensor with the label in the range 0..9.
            uint8image: a [height, width, depth] uint8 Tensor with the image data
    """

    class CIFAR10Record(object):
        pass
    result = CIFAR10Record()

    # Dimensions of the images in the CIFAR-10 dataset.
    # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
    # input format.
    label_bytes = 1  # 2 for CIFAR-100
    result.height = 32
    result.width = 32
    result.depth = 3
    image_bytes = result.height * result.width * result.depth
    # Every record consists of a label followed by the image, with a
    # fixed number of bytes for each.
    record_bytes = label_bytes + image_bytes

    # Read a record, getting filenames from the filename_queue.  No
    # header or footer in the CIFAR-10 format, so we leave header_bytes
    # and footer_bytes at their default of 0.
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(filename_queue)

    # Convert from a string to a vector of uint8 that is record_bytes long.
    record_bytes = tf.decode_raw(value, tf.uint8)

    # The first bytes represent the label, which we convert from uint8->int32.
    result.label = tf.cast(
            tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)

    # The remaining bytes after the label represent the image, which we reshape
    # from [depth * height * width] to [depth, height, width].
    depth_major = tf.reshape(
            tf.strided_slice(record_bytes, [label_bytes],
                                             [label_bytes + image_bytes]),
            [result.depth, result.height, result.width])
    # Convert from [depth, height, width] to [height, width, depth].
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])

    return result


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
    """Construct a queued batch of images and labels.

    Args:
        image: 3-D Tensor of [height, width, 3] of type.float32.
        label: 1-D Tensor of type.int32
        min_queue_examples: int32, minimum number of samples to retain
            in the queue that provides of batches of examples.
        batch_size: Number of images per batch.
        shuffle: boolean indicating whether to use a shuffling queue.

    Returns:
        images: Images. 4D tensor of [batch_size, height, width, 3] size.
        labels: Labels. 1D tensor of [batch_size] size.
    """
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    num_preprocess_threads = 8
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
                [image, label],
                batch_size=batch_size,
                num_threads=num_preprocess_threads,
                capacity=min_queue_examples + 3 * batch_size,
                min_after_dequeue=min_queue_examples)
    else:
        images, label_batch = tf.train.batch(
                [image, label],
                batch_size=batch_size,
                num_threads=num_preprocess_threads,
                capacity=min_queue_examples + 3 * batch_size)

    # Display the training images in the visualizer.
    tf.summary.image('images', images)

    return images, tf.reshape(label_batch, [batch_size])

def get_inputs(data_dir, batch_size, is_test=False):
    """Construct distorted input for CIFAR training using the Reader ops.

    Args:
        data_dir: Path to the CIFAR-10 data directory.
        batch_size: Number of images per batch.

    Returns:
        images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
        labels: Labels. 1D tensor of [batch_size] size.
    """
    if not is_test:
        filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
                                 for i in xrange(1, 6)]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    else:
        filenames = [os.path.join(data_dir, 'test_batch.bin')]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filenames)

    # Read examples from files in the filename queue.
    read_input = read_cifar10(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    # Image processing for training the network. Note the many random
    # distortions applied to the image.

    # Randomly crop a [height, width] section of the image.


    # Because these operations are not commutative, consider randomizing
    # the order their operation.
    # NOTE: since per_image_standardization zeros the mean and makes
    # the stddev unit, this likely has no effect see tensorflow#1458.
    if not is_test:
        # Randomly flip the image horizontally.
        distorted_image = tf.random_crop(reshaped_image, [height, width, 3])
        distorted_image = tf.image.random_flip_left_right(distorted_image)
        # distorted_image = tf.image.random_brightness(distorted_image,max_delta=63)
        # distorted_image = tf.image.random_contrast(distorted_image,lower=0.2, upper=1.8)
        #distorted_image = distorted_image + tf.random_normal(shape=tf.shape(distorted_image), mean=0.0, stddev=0.2, dtype=tf.float32) 
    else:
        distorted_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, height, width)

    # DON'T WANT TO DO THIS SINCE THE OUTPUT FROM THE GENERATOR IS SIGMOID
    # Subtract off the mean and divide by the variance of the pixels.
    # float_image = tf.image.per_image_standardization(distorted_image)
    
    #float_image = tf.multiply(distorted_image,1.0/255.0)
    float_image = tf.multiply(distorted_image,1.0/128.0)
    float_image = tf.add(float_image,-1.0)


    # Set the shapes of tensors.
    float_image.set_shape([height, width, 3])
    read_input.label.set_shape([1])

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                                                     min_fraction_of_examples_in_queue)
    print ('Filling queue with %d CIFAR images before starting to train. '
                 'This will take a few minutes.' % min_queue_examples)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(float_image,read_input.label,min_queue_examples,batch_size,shuffle=True)

def train_inputs(batch_size):
    data_dir = '/u/training/tra016/scratch/Personal/cifar10_data/cifar-10-binary/cifar-10-batches-bin'
    images, labels = get_inputs(data_dir=data_dir,batch_size=batch_size)
    return images, labels

def test_inputs(batch_size):
    data_dir = '/u/training/tra016/scratch/Personal/cifar10_data/cifar-10-binary/cifar-10-batches-bin'
    images, labels = get_inputs(data_dir=data_dir,batch_size=batch_size, is_test=True)
    return images, labels

def plot(samples):
    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(8, 8)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample)
    return fig

def plot_weights(samples):
    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(8, 8)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        #ax.set_aspect('equal')
        plt.imshow(sample)
    return fig

def plot_classes(samples):
    fig = plt.figure(figsize=(10, 100))
    gs = gridspec.GridSpec(1, 10)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        #ax.set_aspect('equal')
        plt.imshow(sample)
    return fig

batch_size = 64

with tf.device('/cpu:0'):
    images, labels = train_inputs(batch_size)
    images_test, labels_test = test_inputs(batch_size)

with tf.variable_scope('placeholder'):
    X = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, 3])
    z = tf.placeholder(tf.float32, [None, 100])  # noise
    y = tf.placeholder(name='label',dtype=tf.float32,shape=[batch_size,10])
    keep_prob = tf.placeholder(tf.float32 ,shape=())
    is_train = tf.placeholder(tf.bool ,shape=())

with tf.variable_scope('GAN'):
    G = generator(z, keep_prob=keep_prob, is_train=is_train)

    D_real, D_real_logits, flat_features = discriminator(X,
        keep_prob=keep_prob, is_train=is_train, reuse=False)
    D_fake, D_fake_logits, flat_features = discriminator(G,
        keep_prob=keep_prob, is_train=is_train, reuse=True)
with tf.variable_scope('D_loss'):
    real_label = tf.concat([0.89*y,tf.zeros([batch_size,1])],axis=1)
    real_label += 0.01*tf.ones([batch_size,11])

    fake_label = tf.concat([tf.zeros([batch_size,10]),
        tf.ones([batch_size,1])],axis=1)

    d_loss_real = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
        logits=D_real_logits,labels=real_label))
    
    d_loss_fake = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
        logits=D_fake_logits,labels=fake_label))

    d_loss = d_loss_real + d_loss_fake

with tf.variable_scope('G_loss'):
    g_loss = tf.reduce_mean(tf.log(D_fake[:,-1]))

with tf.variable_scope('accuracy'):
    correct_prediction = tf.equal(
        tf.argmax(D_real[:,:-1],1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(
        tf.cast(correct_prediction,tf.float32))

tvar = tf.trainable_variables()
dvar = [var for var in tvar if 'discriminator' in var.name]
gvar = [var for var in tvar if 'generator' in var.name]

with tf.name_scope('train'):
    d_train_step = tf.train.AdamOptimizer(
        learning_rate=0.5*(1e-4), beta1=0.5).minimize(d_loss, var_list=dvar)
    g_train_step = tf.train.AdamOptimizer(
        learning_rate=1e-4, beta1=0.5).minimize(g_loss, var_list=gvar)

same_input = np.random.uniform(-1., 1., [64, 100])
same_input = same_input/np.sqrt(np.sum(same_input**2,axis=1))[:,None]

sess = tf.InteractiveSession()

init = tf.global_variables_initializer()
sess.run(init)

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord)
tf.train.start_queue_runners()

num_img = 0
d_loss_real_p = 0
d_loss_fake_p = 0
g_loss_p = 0
t = time.time()
for i in range(0,250000):
    # update D
    X_batch, labels_batch = sess.run([images, labels])
    z_batch = np.random.uniform(-1., 1., [batch_size, 100])
    z_batch = z_batch/np.sqrt(np.sum(z_batch**2,axis=1))[:,None]
    y_batch = np.zeros((batch_size,10))
    y_batch[np.arange(batch_size),labels_batch] = 1

    _, d_loss_real_p, d_loss_fake_p, accuracy_p = sess.run(
        [d_train_step, d_loss_real, d_loss_fake, accuracy], 
        feed_dict={X: X_batch, z: z_batch, 
        y: y_batch, keep_prob:0.5,  is_train:True})

    # update G
    z_batch = np.random.uniform(-1., 1., [batch_size, 100])
    z_batch = z_batch/np.sqrt(np.sum(z_batch**2,axis=1))[:,None]
    _, g_loss_p = sess.run([g_train_step, g_loss], 
        feed_dict={X: X_batch, z: z_batch, keep_prob:0.5, is_train:True})

    # monitor progress
    if i % 20 == 0:
        print('time: %f epoch:%d g_loss:%f d_loss_real:%f d_loss_fake:%f accuracy:%f'
            % (float(time.time()-t), i, g_loss_p, 
            d_loss_real_p, d_loss_fake_p, accuracy_p))
        t = time.time()

    # every 500 batches, save the generator output
    if i % 500 == 0:
        samples = sess.run(G, 
            feed_dict={z: same_input, keep_prob:1.0, is_train:False})
        samples += 1.0
        samples /= 2.0
        fig = plot(samples)
        plt.savefig('output/%s.png' % str(num_img).zfill(3), bbox_inches='tight')
        num_img += 1
        plt.close(fig)

    # every 500 batches, try the test dataset
    if i % 500 == 0:
        test_accuracy = 0.0
        accuracy_count = 0
        for j in xrange(50):
            X_batch, labels_batch = sess.run([images_test,labels_test])
            y_batch = np.zeros((batch_size,10))
            y_batch[np.arange(batch_size),labels_batch] = 1

            accuracy_p = sess.run([accuracy], 
                feed_dict={X: X_batch, y: y_batch, keep_prob:1.0, is_train:False})

            test_accuracy += accuracy_p[0]
            accuracy_count += 1
        test_accuracy = test_accuracy/accuracy_count
        print('TEST:%f' % test_accuracy)

all_vars = tf.global_variables()
dvars = [var for var in all_vars if 'discriminator' in var.name]
dvars = [var for var in dvars if 'Adam' not in var.name]
saver = tf.train.Saver(dvars)
saver.save(sess, 'GAN/discriminator/model')

gvars = [var for var in all_vars if 'generator' in var.name]
gvars = [var for var in gvars if 'Adam' not in var.name]
saver = tf.train.Saver(gvars)
saver.save(sess, 'GAN/generator/model')