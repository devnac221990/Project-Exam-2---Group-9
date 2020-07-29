import numpy as np
import os
import string
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.datasets import fetch_20newsgroups
import random as rn
from keras.layers import  Dense, Embedding, LSTM
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
np.random.seed(1)
rn.seed(1)

# Build the corpus and sequences
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
STOPWORDS = stopwords.words('english')
sequenceLength = 200
output_dim = 300


def tokenize (text):        #   no punctuation & starts with a letter & between 2-15 characters in length
    tokens = [word.strip(string.punctuation) for word in RegexpTokenizer(r'\b[a-zA-Z][a-zA-Z0-9]{2,14}\b').tokenize(text)]
    return  [f.lower() for f in tokens if f and f.lower() not in STOPWORDS]

def preProcess20news():
    X, labels, labelToName = [], [], {}
    twenty_news_data = fetch_20newsgroups(subset='all', remove=( 'footers','headers', 'quotes'), shuffle=True, random_state=42)
    for i, article in enumerate(twenty_news_data['data']):
        stopped = tokenize (article)
        if (len(stopped) == 0):
            continue
        groupIndex = twenty_news_data['target'][i]
        X.append(stopped)
        labels.append(groupIndex)
        labelToName[groupIndex] = twenty_news_data['target_names'][groupIndex]
    nTokens = [len(x) for x in X]
    return X, np.array(labels), labelToName, nTokens



X, labels, labelToName, nTokens = preProcess20news()
labelToNameSortedByLabel = sorted(labelToName.items(), key=lambda kv: kv[0]) # List of tuples sorted by the label number [ (0, ''), (1, ''), .. ]
namesInLabelOrder = [item[1] for item in labelToNameSortedByLabel]
numClasses = len(namesInLabelOrder)


tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
encoded_docs = tokenizer.texts_to_sequences(X)
Xencoded = pad_sequences(encoded_docs, maxlen=sequenceLength, padding='post')

# Test & Train Split
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=1).split(Xencoded, labels)
train_indices, test_indices = next(sss)
train_x, test_x = Xencoded[train_indices], Xencoded[test_indices]
train_labels = to_categorical(labels[train_indices], len(labelToName))
test_labels = to_categorical(labels[test_indices], len(labelToName))

early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=2, mode='auto', restore_best_weights=False)
model = Sequential()
embedding = Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=output_dim, input_length=sequenceLength, trainable=True, mask_zero=True)
model.add(embedding)
model.add(LSTM(units=150, dropout=0.2, recurrent_dropout=0.2, return_sequences=False))
model.add(Dense(numClasses, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
print(model.summary())


result = {}
history = model.fit(x=train_x, y=train_labels, epochs=5, batch_size=32, shuffle=True, validation_data = (test_x, test_labels), verbose=2, callbacks=[early_stop])

result['history'] = history.history
result['test_loss'], result['test_accuracy'] = model.evaluate(test_x, test_labels, verbose=2)
predicted = model.predict(test_x, verbose=2)
predicted_labels = predicted.argmax(axis=1)

print (classification_report(labels[test_indices], predicted_labels, digits=4, target_names=namesInLabelOrder))



import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

















#result['confusion_matrix'] = confusion_matrix(labels[test_indices], predicted_labels).tolist()
#result['classification_report'] = classification_report(labels[test_indices], predicted_labels, digits=4, target_names=namesInLabelOrder, output_dict=True)
#print (confusion_matrix(labels[test_indices], predicted_labels))