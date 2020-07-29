#import the libraries to read the dataset and execute the program
import matplotlib as plt
import numpy as np
import pandas as pd

from keras.datasets import cifar10
from sklearn.decomposition import PCA
#training and testing data from cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
#print the shape of training and testing dataset
print('Traning data shape:', x_train.shape)
print('Testing data shape:', x_test.shape)
y_train.shape,y_test.shape
# fetch the unique number of labels from training data
classes = np.unique(y_train)
nClasses = len(classes)
print('Total number of outputs : ', nClasses)
print('Output classes : ', classes)

import matplotlib.pyplot as plt

label_dict = {
 0: 'airplane',
 1: 'automobile',
 2: 'bird',
 3: 'cat',
 4: 'deer',
 5: 'dog',
 6: 'frog',
 7: 'horse',
 8: 'ship',
 9: 'truck',
}

plt.figure(figsize=[5,5])

# To plot and display the image from the dataset
plt.subplot(121)
curr_img = np.reshape(x_train[0], (32,32,3))

print(plt.title("(Label: " + str(label_dict[y_train[0][0]]) + ")"))
plt.imshow(curr_img)
plt.show()
# to plot and display the image from test dataset
plt.subplot(122)
curr_img = np.reshape(x_test[0],(32,32,3))

print(plt.title("(Label: " + str(label_dict[y_test[0][0]]) + ")"))
plt.imshow(curr_img)
plt.show()
np.min(x_train),np.max(x_train)
#scaling the images
x_train = x_train/255.0
np.min(x_train),np.max(x_train)

x_train.shape
#to reshape the images
x_train_flat = x_train.reshape(-1,3072)
feat_cols = ['pixel'+str(i) for i in range(x_train_flat.shape[1])]
df_cifar = pd.DataFrame(x_train_flat,columns=feat_cols)
df_cifar['label'] = y_train
print('Size of the dataframe: {}'.format(df_cifar.shape))

print(df_cifar.head())
pca_cifar = PCA(n_components=2)
principalComponents_cifar = pca_cifar.fit_transform(df_cifar.iloc[:,:-1])
principal_cifar_Df = pd.DataFrame(data = principalComponents_cifar
             , columns = ['principal component 1', 'principal component 2'])
principal_cifar_Df['y'] = y_train
principal_cifar_Df.head()
print('Explained variation per principal component: {}'.format(pca_cifar.explained_variance_ratio_))

import seaborn as sns
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="principal component 1", y="principal component 2",
    hue="y",
    palette=sns.color_palette("hls", 10),
    data=principal_cifar_Df,
    legend="full",
    alpha=0.3
)
plt.show()

x_test = x_test/255.0
x_test = x_test.reshape(-1,32,32,3)

x_test_flat = x_test.reshape(-1,3072)
pca = PCA(0.9)
pca.fit(x_train_flat)
PCA(copy=True, iterated_power='auto', n_components=0.9, random_state=None,
  svd_solver='auto', tol=0.0, whiten=False)
pca.n_components_
train_img_pca = pca.transform(x_train_flat)
test_img_pca = pca.transform(x_test_flat)
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.optimizers import RMSprop
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
batch_size = 128
num_classes = 10
epochs = 20
# create the sequential model and feed the dataset to the model
model = Sequential()
model.add(Dense(1024, activation='relu', input_shape=(99,)))
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(train_img_pca, y_train,batch_size=batch_size,epochs=epochs,verbose=1,
                    validation_data=(test_img_pca, y_test))

