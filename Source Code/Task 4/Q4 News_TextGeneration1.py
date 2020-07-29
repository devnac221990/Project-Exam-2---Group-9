# project 2 - task 4
import re
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import keras.utils as ku 

# seed for reproducability
from tensorflow import set_random_seed
#import tensorflow as tf
from numpy.random import seed
set_random_seed(2)
#tf.random.set_seed(2)
seed(1)

import pandas as pd
import numpy as np
import string, os
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

def clean_text(txt):
    # Clean text from punctuation [!"#$%&'()*+, -./:;<=>?@[\]^_`{|}~] and convert the words to all lower case
    txt = "".join(v for v in txt if v not in string.punctuation).lower()
    txt = txt.encode("utf8").decode("ascii",'ignore')
    return txt 

def create_model(max_sequence_len, total_words):
    input_len = max_sequence_len - 1
    model = Sequential()
    
    # add embedding Layer
    model.add(Embedding(total_words, 15, input_length=input_len))
    
    # add LSTM Layer as hidden Layer 1
    model.add(LSTM(196))
    model.add(Dropout(0.1))

    model.add(Dense(total_words, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    return model

def generate_text(seed_text, next_words, model, max_sequence_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict_classes(token_list, verbose=0)
        
        output_word = ""
        for word,index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " "+output_word
    return seed_text.title()

# Path for newyork_headline folder that contains 2 type of files Articles & Comments
filepath = r'C:\Users\badri\PycharmProjects\project Exam2\newyork_headline\\'

# Create headlines list
all_headlines = []
# # Read the Articles files only by verifying the file name have 'Articles'
# for filename in os.listdir(filepath):
#     if 'Articles' in filename:
#         # convert file to DataFrame
#         article_df = pd.read_csv(filepath + filename)
#         # returns a list of all headlines in the file
#         all_headlines.extend(list(article_df.headline.values))
# Read all .csv files in the folder, i stopped this one because pc will take to much time
# to handle all the data
for filename in os.listdir(filepath):
    if re.search('\.csv', filename):
        # convert file to DataFrame
        headline_df = pd.read_csv(filepath + filename)
        # returs a list of all headlines in the file
        if 'headline' in headline_df.columns:
            all_headlines.extend(list(headline_df.headline.values))
        elif 'commentBody' in headline_df.columns:
            all_headlines.extend(list(headline_df.commentBody.values))

# Cleanup Unknown headlines from the list
all_headlines = [h for h in all_headlines if h != "Unknown"]
# to find the total number of headlines that we have
print('all_headlins list length = ',len(all_headlines))

# Cleanup the Text in Dataset or headlines list 
headlines = [clean_text(x) for x in all_headlines]
# print 10 headlines ex; ['nfl vs politics has been battle all season long', 'voice vice veracity', 'a standups downward slide',...
print(headlines[:10])

# Tokenize  the headlines
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_headlines)
total_words = len(tokenizer.word_index) + 1
    
# convert headlines list to sequence
input_sequences = []
for line in headlines:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_sequence = token_list[:i+1]
        input_sequences.append(n_sequence)

# print first 10 sequences ex; [[123, 97], [123, 97, 78], [123, 97, 78, 677], [123, 97, 78, 677, 678], [123, 97, 78, 677, 678, 40], [123, 97, 78, 677, 678, 40, 27],...
print(input_sequences[:10])

# padding the input sequence to get predictor and label
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
    
predictors = input_sequences[:,:-1]
label = input_sequences[:,-1]
label = ku.to_categorical(label, num_classes=total_words)


# call create_model function to create the Sequential model
model = create_model(max_sequence_len, total_words)
model.summary()
# Fitting the model
model.fit(predictors, label, epochs=50, verbose=5)

# User input to enter word or words and then predicate the headline
word = input('Please Enter word\s:')
pword = word.strip()
nextwords = input('Please Enter number of words in Headline:')
nextwords =int(nextwords.strip())
print (generate_text(pword, nextwords, model, max_sequence_len))
