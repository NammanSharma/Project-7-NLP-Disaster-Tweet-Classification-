import streamlit as st
import base64
import joblib
import re

# Load model and vectorizer
model = joblib.load("C:\Users\HP\NextHikes Projects")
vectorizer = joblib.load("vectorizer.pkl")

# Set page configuration

st.set_page_config(
    page_title="Disaster Tweet Classification",
    page_icon= "🌪️",
    layout="wide",
)

# Set background image using base64

def add_bg_image(image_path):
    with open(image_path, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp{{
            background-image: url("data:image/jpg;base64,{encoded}");
            backgroud-size: cover;
            background-attachment:fixed;
            background-repeat: no-repeat;
            }}

            </style>
            """,
            unsafe_allow_html=True
    )

    # Add your background image (replace "backgroung.jpg with your file")
    add_bg_image("background.jpg")

    # Custom CSS styles
st.markdown("""
    <style>
        .main {
            font-family: 'Segoe UI', sans-serif;
        }
        .title-box {
            background-color: rgba(0, 0, 0, 0.6);
            color: #ffffff;
            padding: 30px;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.3);
        }
        .title-box h1 {
            font-size: 44px;
            margin-bottom: 10px;
        }
         }
        .title-box p {
            font-size: 18px;
            color: #dddddd;
        }
        .stTextArea textarea {
            font-size: 16px;
            padding: 12px;
        }
        .stButton > button {
            background-color: #e63946;
            color: white;
            font-weight: bold;
            border-radius: 10px;
            padding: 10px 24px;
            font-size: 18px;
        }
        .stButton > button:hover {
            background-color: #d62828;
            color: white;     
}
        .result-box {
            background-color: rgba(255, 255, 255, 0.85);
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Title Box
st.markdown("""
    <div class="title-box">
            <h1>🌋 Disaster Tweet Classification</h1>
            <p>Instantly detect if a tweet is about a real disaster or not</p>
        </div>
    """,unsafe_allow_html=True)

# Tweet input section
col1, col2, col3 = st.columns([1,2,1])
with col2:
    tweet = st.text_area("Paste your tweet below:", height=180)

    if st.button(" Classify Tweet"):
        if not tweet.strip():
            st.warning("Please enter a tweet to classify.")
        else:
            # Text cleaning
            def clean_text(text):
                text = text.lower()
                text = re.sub(r"http\S+|www\S+|https\S+", '', text)
                text = re.sub(r"@\w+|#\w+", '', text)
                text = re.sub(r"[^\w\s]", '', text)
                return text.strip()    

            cleaned = clean_text(tweet)
            vector = vectorizer.transform([cleaned])
            pred = model.predict(vector)[0]
            proba = model.predict_proba(vector)[0][pred] if hasattr(model, "predict_proba") else None

            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            st.markdown("### 🔎 Prediction Result:")
            if pred == 1:
                st.error(f"🚨 **Disaster Tweet Detected!**\n\nConfidence: `{proba:.2%}`" if proba else "🚨 **Disaster Tweet Detected!**")
            else:
                st.success(f"✅ **Not a Disaster Tweet**\n\nConfidence: `{proba:.2%}`" if proba else "✅ **Not a Disaster Tweet**")
            st.markdown('</div>', unsafe_allow_html=True)

 # Footer
st.markdown("---")
st.markdown("<center>🌍 Built with ❤️ using Streamlit | NLP & Machine Learning</center>", unsafe_allow_html=True)    
   