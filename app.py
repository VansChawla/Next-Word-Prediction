import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------------------------------------------------------
# 1. Load Artifacts (Cached to avoid reloading on every UI interaction)
# -------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    try:
        # Load the LSTM model with the corrected filename
        model = load_model('lstm_model.h5')
        
        # Load the tokenizer
        with open('tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
            
        # Load the maximum sequence length used during training
        with open('max_len.pkl', 'rb') as f:
            max_len = pickle.load(f)
            
        return model, tokenizer, max_len
    except Exception as e:
        st.error(f"Error loading files: {e}")
        return None, None, None

model, tokenizer, max_len = load_artifacts()

# -------------------------------------------------------------------------
# 2. Prediction Function
# -------------------------------------------------------------------------
def predict_next_words(model, tokenizer, text, max_len, num_words=1):
    """Predicts the next 'num_words' based on the input text."""
    predicted_text = text
    
    for _ in range(num_words):
        # Convert text to sequence of integers
        sequence = tokenizer.texts_to_sequences([predicted_text])[0]
        
        # Pad the sequence
        padded_sequence = pad_sequences([sequence], maxlen=max_len-1, padding='pre')
        
        # Predict the next word probabilities
        predicted_probs = model.predict(padded_sequence, verbose=0)
        predicted_word_index = np.argmax(predicted_probs, axis=-1)[0]
        
        # Find the word corresponding to the predicted index
        predicted_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_word_index:
                predicted_word = word
                break
                
        # Append the predicted word to the text for the next iteration
        if predicted_word:
            predicted_text += " " + predicted_word
        else:
            break
            
    return predicted_text

# -------------------------------------------------------------------------
# 3. Streamlit UI
# -------------------------------------------------------------------------
st.title("Next Word Prediction App 🧠")
st.write("Enter a phrase, and the LSTM model will predict the next word(s).")

if model and tokenizer and max_len:
    # User Input
    input_text = st.text_input("Enter your text here:", placeholder="Type something...")
    
    # Allow user to predict multiple words in a row
    num_words_to_predict = st.slider("Number of words to predict:", min_value=1, max_value=10, value=1)
    
    if st.button("Predict"):
        if input_text.strip() == "":
            st.warning("Please enter some text to get a prediction.")
        else:
            with st.spinner("Predicting..."):
                result = predict_next_words(model, tokenizer, input_text, max_len, num_words_to_predict)
                
                st.success("Prediction Complete!")
                st.write("### Result:")
                # Highlight the predicted portion
                original_len = len(input_text)
                st.markdown(f"{input_text} **{result[original_len:]}**")
else:
    st.error("Please ensure 'lstm_model.h5', 'tokenizer.pkl', and 'max_len.pkl' are in the same directory as this script.")