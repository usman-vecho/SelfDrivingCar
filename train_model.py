import numpy as np
from alexnet import alexnet
from tensorflow.keras.callbacks import TensorBoard
from random import shuffle
import cv2
import pandas
import tensorflow as tf

WIDTH = 150
HEIGHT = 150

gpu_options = tf.GPUOptions(allow_growth=True)
session = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

MODEL_NAME = 'Last_model.model'
tensorboard = TensorBoard(log_dir = 'logs_Last_model'.format(MODEL_NAME))
train_data = np.load('Car-dataset/Final_data_balanced.npy', allow_pickle=True)

shuffle(train_data)
shuffle(train_data)

train = train_data[:-150]
test = train_data[-150:]

print(len(np.array([i[0] for i in train])))
print(len(np.array([i[0] for i in train]).reshape(-1,WIDTH,HEIGHT,3)))
#print(len(np.array([i[1] for i in train])))

X = np.array([i[0] for i in train]).reshape(-1,WIDTH,HEIGHT,3)
Y = [i[1] for i in train]

shape = X.shape[1:]
model = alexnet(shape)

X = np.array(X)
Y = np.array(Y)

test_x = np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,3)
test_y = [i[1] for i in test]

test_x = np.array(test_x)
test_y = np.array(test_y)
model.fit(X, Y, batch_size=64, epochs=13 ,validation_data = (test_x,test_y),callbacks = [tensorboard])
model.save(MODEL_NAME)


