#importing libraries for execution
import numpy as np
import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
import re
from keras.utils import to_categorical
import random
import tensorflow
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense,Dropout,Embedding,Conv1D,MaxPooling1D,Flatten
from keras.callbacks import EarlyStopping
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.models import Sequential
from tqdm import tqdm
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='bs4')
lemmatizer = WordNetLemmatizer()
# set the random seed for the session

tensorflow.random.set_seed(123)
random.seed(123)
# Import the training and testing data
train_data = pd.read_csv("train.tsv", sep="\t")
test_data = pd.read_csv("test.tsv", sep="\t")

# print the data head for training and testing dataset
print(train_data.head())
train_data.shape
test_data.head()

# Clean the sentences for processing data
def clean_sentences(df):
    reviews = []
# to remove unnecessary content from the data
    for sent in tqdm(df['Phrase']):

        review_textdata = BeautifulSoup(sent).get_text()

        # syntax to remove the non alphanetical characters from the text
        review_textdata = re.sub("[^a-zA-Z]", " ", review_textdata)

        # Syntax to tokenise the sentences in the dataset
        words = word_tokenize(review_textdata.lower())

        # Syntax to lemmarize the words in the data
        lemma_words = [lemmatizer.lemmatize(i) for i in words]

        reviews.append(lemma_words)
# return the dataset with the above functions applied
    return (reviews)


# The dataset with the clean text is retrieved for further implementation of the model
train_sentences = clean_sentences(train_data)
test_sentences = clean_sentences(test_data)
print(len(train_sentences))
print(len(test_sentences))

tar=train_data.Sentiment.values
y_target=to_categorical(tar)
num_classes=y_target.shape[1]

X_train,X_val,y_train,y_val=train_test_split(train_sentences,y_target,test_size=0.2,stratify=y_target)

#Initialization needed for tokenization

unique_words = set()
len_max = 0

for sent in tqdm(X_train):

    unique_words.update(sent)

    if (len_max < len(sent)):
        len_max = len(sent)

# to fetch the number of uniques words and get it printed
print(len(list(unique_words)))
print(len_max)

tokenizer = Tokenizer(num_words=len(list(unique_words)))
tokenizer.fit_on_texts(list(X_train))
X_train = tokenizer.texts_to_sequences(X_train)
X_val = tokenizer.texts_to_sequences(X_val)
X_test = tokenizer.texts_to_sequences(test_sentences)
# we need to make sure to equalise the padding for all the different lengths of the review in the dataset for CNN model.




X_train = sequence.pad_sequences(X_train, maxlen=len_max)
X_val = sequence.pad_sequences(X_val, maxlen=len_max)
X_test = sequence.pad_sequences(X_test, maxlen=len_max)
print(X_train.shape,X_val.shape,X_test.shape)

early_stopping = EarlyStopping(min_delta = 0.001, mode = 'max', monitor='val_accuracy', patience = 2)
callback = [early_stopping]

#Model using Keras CNN
model=Sequential()
model.add(Embedding(len(list(unique_words)),300,input_length=len_max))
model.add(Conv1D(64,5,activation= 'tanh'))
model.add(Dropout(0.5))
model.add(Conv1D(64,5,activation= 'tanh'))
model.add(Dropout(0.5))
model.add(Dense(100,activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling1D(pool_size=4))
model.add(Flatten())
model.add(Dense(num_classes,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.005),
              metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train, validation_data=(X_val, y_val),epochs=10, batch_size=256, verbose=1, callbacks=callback)


# Create count of the number of epochs
epoch_count = range(1, len(history.history['loss']) + 1)

# Visualize learning curve.
plt.plot(epoch_count, history.history['loss'], 'r--')
plt.plot(epoch_count, history.history['val_loss'], 'b-')
plt.legend(['Training Loss', 'Validation Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()