import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('wordnet')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

# Initialize the WordNet lemmatizer for text preprocessing
lemmatizer = WordNetLemmatizer()

# Load the intents JSON file containing patterns and responses
intents = json.loads(open('intents.json').read())

# Initialize empty lists and variables for words, classes, and documents
words = []
classes = []
documents = []
ignore_letters = ['?', '!', ',', '.']

# Preprocess the patterns and build the words and documents lists
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize words to their base form, remove ignore_letters, and create a set of unique words and classes
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))
classes = sorted(set(classes))

# Save the words and classes as Pickle files for later use
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Prepare the training data and output labels
training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    
    # Create a bag of words representation for each pattern
    for word in words:
        if word in word_patterns:
            bag.append(1)
        else:
            bag.append(0)

    # Create the output label for the intent
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    
    training.append([bag, output_row])

# Shuffle the training data for better training performance
random.shuffle(training)

# Find the maximum length of patterns for padding
max_length = max(len(sample[0]) for sample in training)

# Pad the arrays to ensure a consistent shape for model input
train_x = np.array([np.pad(sample[0], (0, max_length - len(sample[0]))) for sample in training])
train_y = np.array([sample[1] for sample in training])

# Define a neural network model for training
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Configure the Stochastic Gradient Descent (SGD) optimizer
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model on the training data
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

# Save the trained model
model.save('chatbot_model.model', hist)
print("Model training and saving complete.")
