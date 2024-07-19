# import libraries
import tkinter as tk
from tkinter import ttk
from datetime import datetime
from keras.layers import TextVectorization
import re
import requests
# import tensorflow.strings as tf_strings
import json
import string
from keras.models import load_model
from tensorflow import argmax
from keras.preprocessing.text import tokenizer_from_json
from keras.utils import pad_sequences
import numpy as np
import tensorflow as tf

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

## loading the tokenizers

model = load_model('english_to_french_lstm_model')

#load Tokenizer
with open('english_tokenizer.json') as f:
    data = json.load(f)
    english_tokenizer = tokenizer_from_json(data)
    
with open('french_tokenizer.json') as f:
    data = json.load(f)
    french_tokenizer = tokenizer_from_json(data)
    
    
with open('sequence_length.json') as f:
    max_length = json.load(f)

def pad(x, length=None):
    return pad_sequences(x, maxlen=length, padding='post')

def beam_search_decoder(predictions, beam_width=3, epsilon=1e-10):
    sequences = [[[], 0.0]]  
    # Walk over each step in the sequence
    for row in predictions:
        all_candidates = list()
        # Expand each current candidate
        for seq, score in sequences:
            for j, prob in enumerate(row):
                prob = max(prob, epsilon)  # Ensure prob is non-zero
                candidate = [seq + [j], score - np.log(prob)]
                all_candidates.append(candidate)
        # Order all candidates by score (lowest score first)
        ordered = sorted(all_candidates, key=lambda tup: tup[1])
        # Select k best samples
        sequences = ordered[:beam_width]
    return sequences

def translate_to_french(english_sentence, beam_width=3):
    english_sentence = english_sentence.lower()
    
    # Remove punctuation
    for punct in ['.', '?', '!', ',']:
        english_sentence = english_sentence.replace(punct, '')

    english_sentence = english_tokenizer.texts_to_sequences([english_sentence])
    english_sentence = pad(english_sentence, max_length)
    english_sentence = english_sentence.reshape((-1, max_length))   
    predictions = model.predict(english_sentence)[0]
    
    beam_results = beam_search_decoder(predictions, beam_width)
    
    # selecting the best result from beam search outputs
    best_sequence = beam_results[0][0]
    
    french_sentence = french_tokenizer.sequences_to_texts([best_sequence])[0]
    
    # print("French translation: ", french_sentence)
    
    return french_sentence

def solve():
    input_text = input_entry.get()
    url = "https://translate.googleapis.com/translate_a/single"
    params = {
        'client': 'gtx',
        'sl': 'en',  
        'tl': 'fr',  
        'dt': 't',
        'q': input_text  
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        translation = response.json()[0][0][0]
        translated_sent = f"French: {translation}"
    except Exception as e:
        translated_sent = f"Error: {e}"
    
    result_label.config(text=translated_sent)
    

root = tk.Tk()
root.title("Language Translator")
root.geometry("500x300")

font = ('Helvetica', 14)

input_entry = tk.Entry(root, width=80, font=font)
input_entry.pack(pady=10)

instruction_label = tk.Label(root, text="Enter sentence for English to French translation", wraplength=400, font=font)
instruction_label.pack(pady=10)

translate_button = tk.Button(root, text="Translate", command=solve, font=font)
translate_button.pack(pady=10)

result_label = tk.Label(root, text="", wraplength=400, font=font)
result_label.pack(pady=20)

root.mainloop()