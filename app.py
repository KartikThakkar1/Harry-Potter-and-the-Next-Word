import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import streamlit as st

# loading the model and the tokenizer

model = load_model("lstm_next_word.h5")

with open("tokenizer.pickle",'rb') as handle:
    tokenizer = pickle.load(handle)


#function to predict the next word

def predict_next_word(predictor_model,tokenizer,input_sentence,max_len_sentence):

    #first, tokenize the input sentence
    tokens = tokenizer.texts_to_sequences([input_sentence])[0]
    if len(tokens)>= max_len_sentence:
        tokens = tokens[-(max_len_sentence-1):]
    
    #apply padding
    tokens = pad_sequences([tokens],maxlen=max_len_sentence-1,padding='pre')

    predicted_words_with_probabilities = model.predict(tokens,verbose=True)

    index_of_word_with_highest_chance = np.argmax(predicted_words_with_probabilities,axis=1)

    for word, index in tokenizer.word_index.items():

        if index == index_of_word_with_highest_chance:
            return word
    return None


st.title("Welcome to Harry Potter and what would be the next word?")
input_sequence = st.text_input("Enter a sequence of words. Make sure to use some HP words for exciting results.","Now will you drink the polyjuice")

#actions for when the button is pressed:

if st.button("Predict the next word!!"):
    max_len_sentence = model.input_shape[1]+1
    prediction = predict_next_word(predictor_model=model,tokenizer=tokenizer,input_sentence=input_sequence,max_len_sentence=max_len_sentence)
    st.write("The next word is: ")
    st.write(prediction)
    st.write(f"'{input_sequence} {prediction}' looks like a cool sentence. Try more!")
