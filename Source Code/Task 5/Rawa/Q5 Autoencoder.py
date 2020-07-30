from keras import Input
from keras.layers import Dense, Input, LSTM, RepeatVector, TimeDistributed, Embedding
from keras.models import Model,Sequential
from keras.datasets import cifar10
from keras.optimizers import RMSprop
from keras.preprocessing import sequence
from numpy import array
from keras.utils import np_utils, plot_model

# this is the size of our encoded representations
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
input_img = Input(shape=(3072,))


# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)

# "h_encoded1" is the hidden encoded representation of the encoded result
h_encoded1 = Dense(encoding_dim, activation='relu')(encoded)

# "decoded" is the lossy reconstruction of the input
decoded1 = Dense(3072, activation='sigmoid')(h_encoded1)
decoded = Dense(3072, activation='sigmoid')(decoded1)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)
# this model maps an input to its encoded representation
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


import numpy as np
from keras import datasets

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# normalize inputs from 0-255 to 0.0-1.0
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train / 255.0
x_test = x_test / 255.0
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

data = autoencoder.fit(x_train, x_train,
                epochs=3,
                batch_size=128,
                shuffle=True,validation_data=(x_test,x_test))

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
model = Sequential()
model.add(Dense(1024, activation='relu', input_shape=(3072,)))
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
print(model.summary())

model.compile(loss='categorical_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])
print('accuracy','loss')


history = model.fit(x_train, y_train,batch_size=128,epochs=3,verbose=1,
                  validation_data=(x_test, y_test))
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Autoencoder/model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='lower right')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Auoencoder/model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper right')
plt.show()

