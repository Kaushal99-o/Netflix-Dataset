import pickle

# Loading the trained model
with open("netflix_model.pkl", "rb") as f:
    model = pickle.load(f)

# Loading the encoders
with open("le_country.pkl", "rb") as f:
    le_country = pickle.load(f)

with open("le_genre.pkl", "rb") as f:
    le_genre = pickle.load(f)

with open("le_type.pkl", "rb") as f:
    le_type = pickle.load(f)





import pickle
import streamlit as st
import numpy as np

# Load your model and encoders
with open("netflix_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("le_country.pkl", "rb") as f:
    le_country = pickle.load(f)
with open("le_genre.pkl", "rb") as f:
    le_genre = pickle.load(f)
with open("le_type.pkl", "rb") as f:
    le_type = pickle.load(f)

# Streamlit app
import streamlit as st
st.title("Netflix Title Type Prediction App")

country_input = st.text_input("Enter Country")
genre_input = st.text_input("Enter Genre Listed In")

if st.button("Predict"):
    # Transform inputs as arrays
    country_encoded = le_country.transform([country_input])
    genre_encoded = le_genre.transform([genre_input])

    # Make proper 2D array for model
    X = np.array([[country_encoded[0], genre_encoded[0]]])

    # Predict
    prediction = model.predict(X)
    predicted_type = le_type.inverse_transform(prediction)[0]

    st.success(f"The predicted Netflix title type is: {predicted_type}")  #In scikit-learn, inverse_transform basically decodes labels back to their original human-readable form.

 #& &"D:\anaconda\envs\Basics\python.exe" -m streamlit run "D:/Basics/datasets/PROJECT/for streamlit.py" --server.headless true --browser.gatherUsageStats false


#Please run the above code for streamlit output