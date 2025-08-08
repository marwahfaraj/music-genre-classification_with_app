import streamlit as st
from pathlib import Path
from PIL import Image
import time
import pandas as pd
import joblib
import numpy as np

st.set_page_config(
    page_title="Music Genre Classification",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Top Banner Image
banner_path = Path("../images/World-Music-Globe-.png")
if banner_path.exists():
    st.image(str(banner_path), use_container_width=False, width=400)
    st.markdown("**Built by: Marwah Faraj & Niyat Kahsay**")

# Main Title
st.title("Music Genre Classification App")
st.markdown("""
#### Upload song features to predict the genre, or explore model performance on the test set.
""")

# Sidebar
st.sidebar.title("🎶 Music Genre Classifier")
st.sidebar.markdown("""
Welcome! This app predicts the genre of a song based on its features and visualizes model performance.
""")

# Sidebar: About the Model
st.sidebar.markdown("---")
st.sidebar.subheader("About the Model")
st.sidebar.markdown("""
- **Model:** XGBoost Classifier  
- **Features:** 14 audio features from Spotify  
- **Tested on:** 2,000+ tracks  
- **Best performance among all tested models**
""")

# Load model (cache for performance)
@st.cache_resource(show_spinner=False)
def load_model():
    model_path = Path(__file__).parent / "model" / "best_xgboost_model.pkl"
    return joblib.load(model_path)

# Load scaler (cache for performance)
@st.cache_resource(show_spinner=False)
def load_scaler():
    scaler_path = Path(__file__).parent / "model" / "standard_scaler.pkl"
    return joblib.load(scaler_path)

# Load genre encoder (cache for performance)
@st.cache_resource(show_spinner=False)
def load_genre_encoder():
    # Create the same encoder used in training
    from sklearn.preprocessing import LabelEncoder
    genre_encoder = LabelEncoder()
    # Fit with the same classes as in training (in order: edm, latin, pop, r&b, rap, rock)
    genre_encoder.fit(['edm', 'latin', 'pop', 'r&b', 'rap', 'rock'])
    return genre_encoder

# Get feature columns from test set
FEATURES = [
    'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
    'duration_s', 'track_popularity', 'track_album_release_year'
]

def preprocess_input(df):
    scaler = load_scaler()
    df_scaled = scaler.transform(df[FEATURES])
    return pd.DataFrame(df_scaled, columns=FEATURES)

def predict_genre_with_confidence(df, model):
    df_proc = preprocess_input(df)
    proba = model.predict_proba(df_proc[FEATURES])
    preds = model.classes_[np.argmax(proba, axis=1)]
    confidences = np.max(proba, axis=1)
    return preds, confidences

# Navigation
page = st.sidebar.radio(
    "Go to:",
    ("🔍 Predict Genre", "📊 Model Evaluation")
)

if page == "🔍 Predict Genre":
    st.header("Predict Song Genre")
    st.info("Upload a CSV file with song features or enter them manually.")
    model = load_model()

    # --- Batch Prediction (CSV Upload) ---
    st.subheader("Batch Prediction (CSV Upload)")
    uploaded_file = st.file_uploader("Upload song features (CSV)", type=["csv"])
    if uploaded_file:
        with st.spinner("Predicting genres, please wait..."):
            df = pd.read_csv(uploaded_file)
            # Check for required columns
            missing_cols = [col for col in FEATURES if col not in df.columns]
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
            else:
                st.subheader("Debug: Input Features for Prediction")
                st.dataframe(df[FEATURES].head())
                preds, confidences = predict_genre_with_confidence(df, model)
                genre_encoder = load_genre_encoder()
                pred_genres = genre_encoder.inverse_transform(preds)
                df_result = df.copy()
                df_result['Predicted Genre'] = pred_genres
                df_result['Confidence (%)'] = (confidences * 100).round(2)
                st.success(f"Predicted genres for {len(df)} songs!")
                st.dataframe(df_result)
                # Download button
                csv = df_result.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Predictions as CSV",
                    data=csv,
                    file_name='genre_predictions.csv',
                    mime='text/csv'
                )
    else:
        st.caption("Upload a CSV file with the required features for batch prediction.")

    # --- Single Song Prediction (Manual Entry) ---
    st.subheader("Single Song Prediction (Manual Entry)")
    with st.form("single_song_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            danceability = st.number_input("Danceability", 0.0, 1.0, 0.5)
            energy = st.number_input("Energy", 0.0, 1.0, 0.5)
            key = st.number_input("Key", 0, 11, 0)
            loudness = st.number_input("Loudness (dB)", -60.0, 0.0, -10.0)
            mode = st.selectbox("Mode", [0, 1], format_func=lambda x: "Minor" if x == 0 else "Major")
        with col2:
            speechiness = st.number_input("Speechiness", 0.0, 1.0, 0.05)
            acousticness = st.number_input("Acousticness", 0.0, 1.0, 0.1)
            instrumentalness = st.number_input("Instrumentalness", 0.0, 1.0, 0.0)
            liveness = st.number_input("Liveness", 0.0, 1.0, 0.1)
            valence = st.number_input("Valence", 0.0, 1.0, 0.5)
        with col3:
            tempo = st.number_input("Tempo (BPM)", 0.0, 300.0, 120.0)
            duration_s = st.number_input("Duration (s)", 0.0, 600.0, 180.0)
            track_popularity = st.number_input("Track Popularity", 0, 100, 50)
            track_album_release_year = st.number_input("Release Year", 1900, 2025, 2020)
        submitted = st.form_submit_button("Predict Genre")
    if submitted:
        with st.spinner("Predicting genre..."):
            input_dict = {
                'danceability': danceability,
                'energy': energy,
                'key': key,
                'loudness': loudness,
                'mode': mode,
                'speechiness': speechiness,
                'acousticness': acousticness,
                'instrumentalness': instrumentalness,
                'liveness': liveness,
                'valence': valence,
                'tempo': tempo,
                'duration_s': duration_s,
                'track_popularity': track_popularity,
                'track_album_release_year': track_album_release_year
            }
            input_df = pd.DataFrame([input_dict])
            st.subheader("Debug: Input Features for Prediction")
            st.dataframe(input_df[FEATURES])
            pred, conf = predict_genre_with_confidence(input_df, model)
            genre_encoder = load_genre_encoder()
            pred_genre = genre_encoder.inverse_transform(pred)[0]
            st.success(f"Predicted Genre: {pred_genre} (Confidence: {conf[0]*100:.2f}%)")

elif page == "📊 Model Evaluation":
    st.empty()  # Clear any previous content
    st.markdown("---")  # Add separator
    st.header("Model Evaluation on Test Set")
    st.info("Test set metrics and visualizations for the XGBoost model.")
    with st.spinner("Loading evaluation metrics..."):
        time.sleep(1)  # Simulate loading
        st.image(str(Path("../images/models/confusion_matrix (1).png")), caption="Confusion Matrix (XGBoost)", use_container_width=True)
        st.image(str(Path("../images/models/roc_curve_ovr.png")), caption="ROC Curve (XGBoost, One-vs-Rest)", use_container_width=True)

st.markdown("---")
st.caption("Developed for 504 Final Project | Powered by Streamlit")