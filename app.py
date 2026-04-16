import streamlit as st
import joblib
import os

# 1. Load the trained model
MODEL_PATH = os.path.join("model_artifacts", "sms_spam_model.joblib")
model = joblib.load(MODEL_PATH)

# 2. App title
st.title("SMS Spam Detector ")
st.write("Type or paste an SMS message below and see if it is SPAM or HAM.")

# 3. Input text box
user_input = st.text_area("Enter SMS message here:")

# 4. Predict button
if st.button("Check SMS"):
    if user_input.strip() == "":
        st.warning("Please enter a message first!")
    else:
        prediction = model.predict([user_input])[0]
        if prediction == 1:
            st.error(" This message is SPAM!")
        else:
            st.success(" This message is HAM (not spam)")