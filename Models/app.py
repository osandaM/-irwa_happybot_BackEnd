from flask import Flask, request, jsonify
from flask_cors import CORS
import random
import json
import pickle
import numpy as np
import re
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

from database import connect_to_mongodb

app = Flask(__name__)
CORS(app)

# Load the model, words, classes, and intents only once
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.model')

# Clean up the user's sentence for processing
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

# Convert user's sentence into a bag of words
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# Predict the intent of the user's sentence
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

# Get a response based on the predicted intent
def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

# Extract the product name from the user's query (simplified)
def get_product_name(message):
    words = message.split()
    if words:
        product_name = words[-1]
        return product_name
    return None

# Retrieve product information from the database
def retrieve_product_info_from_db(product_name):
    db = connect_to_mongodb()
    collection = db["IoT devices"]
    regex_pattern = re.compile(re.escape(product_name), re.IGNORECASE)
    cursor = collection.find({"deviceName": {"$regex": regex_pattern}})
    results = []

    for document in cursor:
        result_dict = {key: value for key, value in document.items() if key != '_id'}
        results.append(result_dict)

    cursor.close()
    return results

# Handle chat requests
@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')  # Get the user's message from the POST request

    ints = predict_class(user_message)
    product_info_intent = next((i for i in ints if i['intent'] == 'product_info'), None)

    if product_info_intent:
        # Extract the product name from the user's query
        product_name = get_product_name(user_message)
        product_info = retrieve_product_info_from_db(product_name)

        if product_info:
            product_name_from_db = product_info[0]['deviceName']
            product_price_from_db = product_info[0]['price']
            product_des_from_db = product_info[0]['description']
            product_availability_from_db = product_info[0]['availability']

            response_data = {
                'bot_response': f"Thank you for your inquiry. {product_name_from_db} is indeed an excellent choice. It is available at a very competitive price of ${product_price_from_db}. Currently, it is {product_availability_from_db} in stock. Here are some detailed specifications for {product_name_from_db}:\n{product_des_from_db}"
            }
        else:
            response_data = {
                'bot_response': "I couldn't find information about the requested product."
            }
    else:
        res = get_response(ints, intents)
        response_data = {
            'bot_response': res
        }

    return jsonify(response_data)

if __name__ == '__main__':
    app.run()
