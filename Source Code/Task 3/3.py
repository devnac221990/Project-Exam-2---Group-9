import matplotlib.pylab as plt
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Lambda, MaxPool2D, \
    BatchNormalization, Input
from keras.models import Sequential
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from pathlib import Path
from keras.optimizers import Adam
import pandas as pd

data = pd.read_csv("monkey_labels.txt")
print(data)

h = 150 #height
w = 150 #width
channels = 3
batch_size = 32
seed = 1337

trainDirectory = Path('C:\\Users\\nikol\\Desktop\\Tasks\\Data 3\\training\\training')
testDirectory = Path('C:\\Users\\nikol\\Desktop\\Tasks\\Data 3\\validation\\validation')
# Training generator
trainDatageneration = ImageDataGenerator(rescale=1. / 255) #Scaling
trainGenerator = trainDatageneration.flow_from_directory(trainDirectory,
                                                         target_size=(h, w),
                                                         batch_size=batch_size,
                                                         seed=seed,
                                                         class_mode='categorical')

# Test generator
testDatageneration = ImageDataGenerator(rescale=1. / 255)
testGenerator = testDatageneration.flow_from_directory(testDirectory,
                                                       target_size=(h, w),
                                                       batch_size=batch_size,
                                                       seed=seed,
                                                       class_mode='categorical')
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(Adam(lr=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])
history = model.fit_generator(trainGenerator,
                              steps_per_epoch=1027 / batch_size,
                              epochs=4,
                              verbose=1,
                              validation_data=testGenerator,
                              validation_steps=4)

model.summary()
print(history.history.keys())
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.title('Training and validation accuracy')
plt.plot(epochs, acc, 'red', label='Training acc')
plt.plot(epochs, val_acc, 'blue', label='Validation acc')
plt.legend()
plt.figure()
plt.title('Training and validation loss')
plt.plot(epochs, loss, 'red', label='Training loss')
plt.plot(epochs, val_loss, 'blue', label='Validation loss')

plt.legend()
plt.show()