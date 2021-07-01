import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D , ZeroPadding2D
from tensorflow.keras.layers import BatchNormalization

run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)
def alexnet(shape):    

  alexnet = Sequential()
  alexnet.add(Conv2D(96, (11, 11), input_shape=shape,
    padding='same'))
  alexnet.add(BatchNormalization())
  alexnet.add(Activation('relu'))
  alexnet.add(MaxPooling2D(pool_size=(2, 2)))
  
  alexnet.add(Conv2D(256, (5, 5), padding='same'))
  alexnet.add(BatchNormalization())
  alexnet.add(Activation('relu'))
  alexnet.add(MaxPooling2D(pool_size=(2, 2)))
  alexnet.add(Dropout(0.2))

  alexnet.add(ZeroPadding2D((1, 1)))
  alexnet.add(Conv2D(512, (3, 3), padding='same'))
  alexnet.add(BatchNormalization())
  alexnet.add(Activation('relu'))
  alexnet.add(MaxPooling2D(pool_size=(2, 2)))
  alexnet.add(Dropout(0.3))
  ###
  alexnet.add(ZeroPadding2D((1, 1)))
  alexnet.add(Conv2D(1024, (3, 3), padding='same'))
  alexnet.add(BatchNormalization())
  alexnet.add(Activation('relu'))
  alexnet.add(MaxPooling2D(pool_size=(2, 2)))
  alexnet.add(Dropout(0.3))

  alexnet.add(ZeroPadding2D((1, 1)))
  alexnet.add(Conv2D(1024, (3, 3), padding='same'))
  alexnet.add(BatchNormalization())
  alexnet.add(Activation('relu'))
  alexnet.add(MaxPooling2D(pool_size=(2, 2)))

  
  alexnet.add(Flatten())
  
  alexnet.add(Dense(512))
  alexnet.add(BatchNormalization())
  alexnet.add(Activation('relu'))
  alexnet.add(Dropout(0.2))

  
  alexnet.add(Dense(1014))
  alexnet.add(BatchNormalization())
  alexnet.add(Activation('relu'))
  alexnet.add(Dropout(0.3))

  
  alexnet.add(Dense(3))
  alexnet.add(BatchNormalization())
  alexnet.add(Activation('softmax'))

  adam = tf.keras.optimizers.Adam(lr=0.0001)

  alexnet.compile(loss='categorical_crossentropy',
                optimizer=adam,
                metrics=['accuracy'])
                #options = run_opts)


  return alexnet
