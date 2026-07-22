import tensorflow as tf
import pickle

model = tf.keras.models.load_model("next_word.predictor.keras")

with open("tokenizer.pkl","rb") as f:
    tokenizer = pickle.load(f)

from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

with open("max_len.pkl","rb") as f:
    max_len = pickle.load(f)

def predict_next_word(model, tokenizer, text, max_len):

    token_list = tokenizer.texts_to_sequences([text])[0]

    token_list = pad_sequences(
        [token_list],
        maxlen=max_len-1,
        padding="pre"
    )

    prediction = model.predict(token_list, verbose=0)

    predicted_index = np.argmax(prediction, axis=-1)[0]

    return tokenizer.index_word.get(predicted_index, "")

def generate_text(model, tokenizer, text, max_len, n_words):

    for _ in range(n_words):

        next_word = predict_next_word(
            model,
            tokenizer,
            text,
            max_len
        )

        if next_word == "":
            break

        text += " " + next_word

    return text

import streamlit as st

st.title("Next Word Predictor")
seed_text = st.text_input(
    "Enter starting sentence"
)
num_words = st.slider(
    "Words to Generate",
    min_value=1,
    max_value=100,
    value=20
)
if st.button("Generate"):

    result = generate_text(
        model,
        tokenizer,
        seed_text,
        max_len,
        num_words
    )

    st.write(result)
