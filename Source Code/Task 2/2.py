import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import string
from sklearn.metrics import classification_report
from keras.layers import Dense, Embedding, LSTM
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.datasets import fetch_20newsgroups
import random

random.seed(1)

from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

np.random.seed(1)
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

STOPWORDS = stopwords.words('english')
sequenceLength = 200
output_dim = 300


def tokenize(text):  # no punctuation, word starts with a letter  and word is between [2-15] characters in length
    tokens = [word.strip(string.punctuation) for word in
              RegexpTokenizer(r'\b[a-zA-Z][a-zA-Z0-9]{2,14}\b').tokenize(text)]
    return [f.lower() for f in tokens if f and f.lower() not in STOPWORDS]


def preProcess20news():
    X, labels, labelToName = [], [], {}  # X:list labels:list labelToName:dictionary
    twenty_news_data = fetch_20newsgroups(subset='all', remove=('footers', 'headers', 'quotes'), shuffle=True,
                                          random_state=42)

    for i, article in enumerate(twenty_news_data['data']):
        stopped = tokenize(article)
        if (len(stopped) == 0):
            continue
        groupIndex = twenty_news_data['target'][i]
        X.append(stopped)
        labels.append(groupIndex)
        labelToName[groupIndex] = twenty_news_data['target_names'][groupIndex]
    nTokens = [len(x) for x in X]
    return X, nTokens, labelToName, np.array(labels)


X, nTokens, labelToName, labels = preProcess20news()
label_to_name_sorted = sorted(labelToName.items(), key=lambda kv: kv[
    0])  # List of tuples sorted by the label number [ (0, ''), (1, ''), .. ]
namesInLabelOrder = [item[1] for item in label_to_name_sorted]
numberOfClasses = len(namesInLabelOrder)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
encoded_docs = tokenizer.texts_to_sequences(X)
padded_sequences = pad_sequences(encoded_docs, maxlen=sequenceLength, padding='post')
# Splits Data 80% train and 20% test
stratifiedShuffleSplitData = StratifiedShuffleSplit(test_size=0.2, n_splits=1, random_state=1).split(padded_sequences,
                                                                                                     labels)
trainIndices, testIndices = next(stratifiedShuffleSplitData)
Xtrain, Xtest = padded_sequences[trainIndices], padded_sequences[testIndices]
train_labels = to_categorical(labels[trainIndices], len(labelToName))
test_labels = to_categorical(labels[testIndices], len(labelToName))

# Prevents overfitting
earlyStopCallback = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=False, verbose=2, mode='auto',
                                  min_delta=0)
model = Sequential()
embeddingLayer = Embedding(input_dim=len(tokenizer.word_index) + 1, mask_zero=True, output_dim=output_dim,
                           trainable=True, input_length=sequenceLength)
model.add(embeddingLayer)
model.add(LSTM(units=150, dropout=0.2, return_sequences=False, recurrent_dropout=0.2))
model.add(Dense(numberOfClasses, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
print(model.summary())

finalRes = {}
history = model.fit(x=Xtrain, y=train_labels, epochs=5, batch_size=32, shuffle=True,
                    validation_data=(Xtest, test_labels), verbose=2, callbacks=[earlyStopCallback])
finalRes['history'] = history.history
finalRes['test_loss'], finalRes['test_accuracy'] = model.evaluate(Xtest, test_labels, verbose=2)
predicted = model.predict(Xtest, verbose=2)
predicted_labels = predicted.argmax(axis=1)

print(classification_report(labels[testIndices], predicted_labels, digits=4, target_names=namesInLabelOrder))

import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
