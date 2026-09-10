import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tempfile
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from utils.config_parser import load_config
from preprocessing.audio_preprocessor import AudioPreprocessor
from feature_extraction.feature_extractor import FeatureExtractor
from visualization.audio_visualizer import AudioVisualizer
from streamlit_app.app_utils import (
    load_model_for_inference,
    predict_audio_genre,
    create_top3_probability_chart,
)

# Page configuration
st.set_page_config(
    page_title="Folk Music Classifier",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load configuration
@st.cache_data
def get_config():
    return load_config(PROJECT_ROOT / "configs" / "config.yaml")

config = get_config()
preprocessor = AudioPreprocessor(
    target_sr=config["dataset"]["target_sr"],
    mono=config["dataset"]["mono"],
    segment_duration=config["dataset"]["segment_duration"],
)
feature_extractor = FeatureExtractor(config)
visualizer = AudioVisualizer(config)

# Title & Header
st.title("Identification & Classification of Folk Music")
st.markdown(
    "**End-to-End AI System powered by Deep Learning & Audio Signal Processing**"
)
st.divider()

# Sidebar Setup
st.sidebar.header("Model & Input Settings")
selected_model_name = st.sidebar.selectbox(
    "Select Deep Learning Model Architecture:",
    options=["cnn", "cnn_bilstm", "crnn","crnn_specaugment"],
    format_func=lambda x: {"cnn": "2D-CNN Model", "cnn_bilstm": "CNN + BiLSTM Model", "crnn": "CRNN Model","crnn_specaugment": "CRNN + SpecAugment"}[x],
)

input_method = st.sidebar.radio(
    "Choose Audio Input Method:",
    options=["Upload Audio File (WAV/MP3)", "Record Microphone Audio"],
)

st.sidebar.markdown("---")
st.sidebar.subheader("Supported Folk Genres (6)")
for g in config["genres"]:
    st.sidebar.markdown(f"- **{g}**")

# Main Input Section
y_audio = None
sr_audio = config["dataset"]["target_sr"]

col_input, col_info = st.columns([2, 1])

with col_input:
    if input_method == "Upload Audio File (WAV/MP3)":
        uploaded_file = st.file_uploader(
            "Upload an audio sample (.wav or .mp3):", type=["wav", "mp3"]
        )
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            st.audio(tmp_path, format="audio/wav")
            y_raw, _ = preprocessor.load_audio(tmp_path)
            y_audio = preprocessor.pad_or_trim(y_raw)
            Path(tmp_path).unlink(missing_ok=True)

    else:
        st.subheader("Microphone Audio Recorder")
        mic_audio = st.audio_input("Record audio sample directly:")
        if mic_audio is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(mic_audio.getvalue())
                tmp_path = tmp_file.name

            st.audio(tmp_path, format="audio/wav")
            y_raw, _ = preprocessor.load_audio(tmp_path)
            y_audio = preprocessor.pad_or_trim(y_raw)
            Path(tmp_path).unlink(missing_ok=True)

with col_info:
    st.info(
        "**Instructions**:\n"
        "1. Select a model from the sidebar.\n"
        "2. Upload a WAV/MP3 recording or use your microphone.\n"
        "3. Click **Classify Folk Music** to see predictions, confidence scores, and signal features."
    )

if y_audio is not None:
    st.divider()

    # Create Tabs for Analysis
    tab_pred, tab_signals, tab_features = st.tabs(
        [" Genre Prediction", "Signal Visualizations", "Extracted Features"]
    )

    # 1. Prediction Tab
    with tab_pred:
        if st.button(" Classify Folk Music", type="primary", use_container_width=True):
            with st.spinner(f"Loading {selected_model_name.upper()} model and analyzing acoustic features..."):
                model = load_model_for_inference(selected_model_name, config)
                top_genre, confidence, all_probs = predict_audio_genre(y_audio, model, config)

            st.markdown("### Prediction Results")
            c1, c2 = st.columns([1, 2])

            with c1:
                st.metric(
                    label="Predicted Folk Genre",
                    value=top_genre,
                    delta=f"{confidence * 100.0:.1f}% Confidence",
                )
                st.progress(float(confidence))

            with c2:
                fig_chart = create_top3_probability_chart(all_probs)
                st.plotly_chart(fig_chart, use_container_width=True)

            # Full Probability Table
            with st.expander("View Full 6-Class Probability Distribution"):
                df_probs = pd.DataFrame(
                    [{"Genre": k, "Probability (%)": f"{v * 100.0:.2f}%"} for k, v in all_probs.items()]
                )
                st.dataframe(df_probs, use_container_width=True)

    # 2. Signals Tab
    with tab_signals:
        st.subheader("Waveform & Spectral Graphs")
        col_s1, col_s2 = st.columns(2)

        with col_s1:
            st.markdown("**Time-Domain Audio Waveform**")
            fig_wave = visualizer.plot_waveform(y_audio, title="Waveform")
            st.pyplot(fig_wave)
            plt.close(fig_wave)

            st.markdown("**40 MFCC Coefficients**")
            mfcc = feature_extractor.extract_all_features(y_audio)["mfcc"]
            fig_mfcc = visualizer.plot_mfcc(mfcc, title="MFCC")
            st.pyplot(fig_mfcc)
            plt.close(fig_mfcc)

        with col_s2:
            st.markdown("**Mel Spectrogram (dB)**")
            mel_db = feature_extractor.extract_mel_spectrogram(y_audio)
            fig_mel = visualizer.plot_mel_spectrogram(mel_db, title="Mel Spectrogram")
            st.pyplot(fig_mel)
            plt.close(fig_mel)

            st.markdown("**Chroma Pitch Features**")
            chroma = feature_extractor.extract_all_features(y_audio)["chroma"]
            fig_chroma = visualizer.plot_chroma(chroma, title="Chroma Features")
            st.pyplot(fig_chroma)
            plt.close(fig_chroma)

    # 3. Features Tab
    with tab_features:
        st.subheader("13 Acoustic & Spectral Feature Summaries")
        feats = feature_extractor.extract_all_features(y_audio)

        feat_summary = []
        for name, arr in feats.items():
            if isinstance(arr, np.ndarray):
                feat_summary.append({
                    "Feature Name": name,
                    "Shape": str(arr.shape),
                    "Mean": round(float(np.mean(arr)), 4),
                    "Std Dev": round(float(np.std(arr)), 4),
                    "Min": round(float(np.min(arr)), 4),
                    "Max": round(float(np.max(arr)), 4),
                })

        df_feats = pd.DataFrame(feat_summary)
        st.dataframe(df_feats, use_container_width=True)
