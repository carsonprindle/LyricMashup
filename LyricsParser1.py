import numpy as np
import sys
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.preprocessing.text import one_hot
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

#DATA HERE -- find 

#GET STUFF FROM CARSON

#Convert into readable form -- modified from https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py and https://chunml.github.io/ChunML.github.io/project/Creating-Text-Generator-Using-Recurrent-Neural-Network/

text= open("kanye_and_taylor.txt").read()
chars = sorted(list(set(text))) #split into a sorted list of characters
vocab_size = len(chars)
char_size = len(text)
print(chars)
print(vocab_size)
print(char_size)


ix_to_char = {ix:char for ix, char in enumerate(chars)} #create a dictionary of the spot of each character
char_to_ix = {char:ix for ix, char in enumerate(chars)}

#create number of sequences
sequence_cap = 15 #about max length for a song couplet
#below to "End pattern organization from: https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/"
dataX = []
dataY = []
for i in range(0, char_size - sequence_cap, 1):
	seq_in = text[i:i + sequence_cap]
	seq_out = text[i + sequence_cap]
	dataX.append([char_to_ix[char] for char in seq_in])
	dataY.append(char_to_ix[seq_out])
n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)
# reshape X to be [samples, time steps, features]
X =  np.reshape(dataX, (n_patterns, sequence_cap,1))
# normalize
X = X / float(vocab_size)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)
#End pattern organization.

#our own model inspired by previous repos
#model = Sequential()
#model.add(LSTM(365, input_shape=(X.shape[1], X.shape[2])))
#model.add(TimeDistributed(Dense(vocab_size)))
#model.add(Dense(sequence_cap))
#model.add(Activation('softmax'))
#model.compile(loss="categorical_crossentropy", optimizer="nadam")

model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

#from machinelearning site above.....
# define the checkpoint
filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
# fit the model
model.fit(X, y, epochs=20, batch_size=128, callbacks=callbacks_list)
start = np.random.randint(0, len(dataX)-1)
pattern = dataX[start]
print("Seed:")
print("\"", ''.join([ix_to_char[value] for value in pattern]), "\"")
# generate characters
for i in range(1000):
	x = np.reshape(pattern, (1, len(pattern), 1))
	x = x / float(vocab_size)
	prediction = model.predict(x, verbose=0)
	index = np.argmax(prediction)
	result = ix_to_char[index]
	seq_in = [ix_to_char[value] for value in pattern]
	sys.stdout.write(result)
	pattern.append(index)
	pattern = pattern[1:len(pattern)]
print("\nDone.")

