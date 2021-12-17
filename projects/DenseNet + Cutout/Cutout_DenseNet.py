# -*- coding: utf-8 -*-
"""
Task 2 A Regularised DenseNet

Created on Sun Dec 12 21:20:16 2021

@author: 21049846
"""

import tensorflow as tf
from tensorflow.keras import datasets, layers
from tensorflow.keras import Input, Model
from PIL import Image, ImageDraw
import numpy as np
# from matplotlib import pyplot as plt

input_img = Input(shape=(32,32,3))

def bottleneck_layer(last_input):
    
    bn = layers.BatchNormalization()(last_input)
    relu = layers.ReLU()(bn)
    conv = layers.Conv2D(64, (1,1))(relu)
    
    return conv

# Contain a member function dense_block, implementing a specific form of DenseNet
# architecture, each contains 4 convolutional layers. [3]
def dense_block(last_output):
    
    concatenation = [last_output]
    for i in range(4):
        input_unit = layers.concatenate(concatenation)
        bottle_neck = bottleneck_layer(input_unit)
        bn = layers.BatchNormalization()(bottle_neck)
        relu = layers.ReLU()(bn)
        conv = layers.Conv2D(64, (3,3), padding='same')(relu)
        concatenation.append(conv)
        
    return conv

def transition_layer(last_output):
    
    bn = layers.BatchNormalization()(last_output)
    relu = layers.ReLU()(bn)
    conv = layers.Conv2D(64, (1,1), padding='same')(relu)
    maxpool = layers.MaxPooling2D((2,2))(conv)
    
    return maxpool

# Design and implement the new network architecture to use 3 of these dense blocks. [4]
def DenseNet(input_img):
    
    conv = layers.Conv2D(32, (3,3), padding='same')(input_img)
    maxpool = layers.MaxPooling2D((2,2))(conv)
    denselayer1 = dense_block(maxpool)
    translayer1 = transition_layer(denselayer1)
    denselayer2 = dense_block(translayer1)
    translayer2 = transition_layer(denselayer2)
    denselayer3 = dense_block(translayer2)
    # flat = layers.Flatten()(denselayer3)
    globalavgpool = layers.GlobalAveragePooling2D()(denselayer3)
    # softmax = layers.Softmax()(globalavgpool)
    dense1 = layers.Dense(64, activation='relu')(globalavgpool)
    dense2 = layers.Dense(10)(dense1)
    
    return dense2

# Summarise and print your network architecture, e.g. using built-in summary function. [1]
dense_model = Model(input_img, DenseNet(input_img))
dense_model.summary()

# Use square masks with variable size and location. [2]
def square_mask(input_img, length, row_loc, col_loc):
    
    mask = np.ones((32,32), np.float32)
    y1 = np.clip(col_loc - length // 2, 0, 32)
    y2 = np.clip(col_loc + length // 2, 0, 32)
    x1 = np.clip(row_loc - length // 2, 0, 32)
    x2 = np.clip(row_loc + length // 2, 0, 32)
    mask[y1:y2, x1:x2] = 0
    
    mask = tf.convert_to_tensor(mask)
    mask = tf.tile(tf.reshape(mask, (32,32,1)), [1,1,3])
    
    input_img = input_img * mask
    
    return input_img

# Add an additional parameter s, such that the mask size can be uniformly sampled from [0, s]. [3]
def uniform_size_square_mask(s, input_img, row_loc, col_loc):
    
    length = tf.random.uniform(shape=[], minval=0, maxval=s, dtype=tf.int32)
    input_img = square_mask(input_img, length, row_loc, col_loc)
    
    return input_img

# Location should be sampled uniformly in the image space. N.B. care needs to be taken around
# the boundaries, so the sampled mask maintains its size. [3]
def uniform_loc_square_mask(s, input_img):
    
    row_loc = tf.random.uniform(shape=[], minval=0, maxval=32, dtype=tf.int32)
    col_loc = tf.random.uniform(shape=[], minval=0, maxval=32, dtype=tf.int32)
    
    input_img = uniform_size_square_mask(s, input_img, row_loc, col_loc)
    
    return input_img


# import dataset. Cifar-10
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)

# normalization
x_train = x_train / np.amax(x_train)
x_test = x_test / np.amax(x_test)

# Visualise your implementation, by saving to a PNG file “cutout.png”, a montage of 16 images
# with randomly augmented images that are about to be fed into network training. [3]
num_example = 16
im = Image.fromarray(tf.concat([(uniform_loc_square_mask(18, x_train[i,...])*255).numpy().astype(np.uint8) for i in range(num_example)], 1).numpy()) 
im.show()
im.save("cutout.png")

# Add Cutout into the network training. [3]
def Cutout_DenseNet(s, input_img):
    
    cutout_img = uniform_loc_square_mask(s, input_img)    
    dense_cutout = DenseNet(cutout_img)
    
    return dense_cutout

# Train the new DenseNet classification network with Cutout data augmentation. [3]
model = Model(input_img, Cutout_DenseNet(10 ,input_img))
model.summary()

model.compile(optimizer='adam', 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              metrics=['accuracy'])

# Run a training with 10 epochs and save the trained model. [3]
train_model = model.fit(x_train, y_train, epochs=10)

# Submit your trained model within the task folder. [2]
model.save('cutout_densenet_saved_model')

# Report the test set performance in terms of classification accuracy versus the epochs. [2]
epoch = [10, 20, 30, 40, 50]
test_accuracy = []
for j in range(len(epoch)):
    train_model = model.fit(x_train, y_train, epochs=epoch[j])
    test_eval = model.evaluate(x_test, y_test, verbose=1)
    test_accuracy.append(test_eval[1])

test_performance = Image.open("test_perform.png")
test_performance.show()
# test_accuracy = [0.7683, 0.7934, 0.8218, 0.8245, 0.8133]    
"""
Note: matplotlib cannot be used in the task scripts.
plt.plot(epoch, test_accuracy, 'ro')
plt.plot(epoch, test_accuracy, 'b')
plt.xlabel("number of epochs")
plt.ylabel("test set accuracy")
plt.title("test set performance")
plt.show()
"""

# Visualise your results, by saving to a PNG file “result.png”, a montage of 36 test images with
# captions indicating the ground-truth and the predicted classes for each. [3]

# make prediction
num_example = 36
prediction = tf.math.argmax(model.predict(x_test[:num_example,...]), 1).numpy()
# montage of 36 test images
im = Image.fromarray(tf.concat([(x_test[i,...]*255).astype(np.uint8) for i in range(num_example)], 1).numpy())
im = im.resize((int(im.size[0]*3), int(im.size[1]*3)))
for i in range(len(prediction)):
    ImageDraw.Draw(im).text((96*i, 0), "pred:{}, true:{}".format(prediction[i], int(y_test[i])), (0,0,0))
im.show()
im.save("result.png")
