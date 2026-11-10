import csv
import gc
from pathlib import Path

import numpy as np
import soundfile as sf
import streamlit as st
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_ID = "facebook/musicgen-small"

st.set_page_config(page_title="AI Background Music for Short Films", page_icon="🎬", layout="wide")
st.title("🎬 AI Background Music for a Short Film")
st.write("Generate three alternative instrumental soundtracks locally, listen to them, and evaluate them.")

st.info(
    "LOCAL MODE: No Hugging Face API token is required. The first run downloads "
    "the MusicGen model (several GB) and needs internet access. CPU generation can be slow."
)

@st.cache_resource(show_spinner="Loading MusicGen Small model...")
def load_model():
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = MusicgenForConditionalGeneration.from_pretrained(MODEL_ID)
    model.to("cpu")
    model.eval()
    return processor, model

def build_prompt(style, mood, pacing, scene, instruments):
    return (
        f"{style}. Instrumental cinematic background score. Emotion: {mood}. "
        f"Scene pacing: {pacing}. Scene: {scene}. Instruments: {instruments}. "
        "No vocals, no singing, no speech, no sound effects."
    )

def save_audio(audio_tensor, sample_rate, path):
    audio = np.squeeze(audio_tensor.detach().cpu().float().numpy())
    if audio.ndim not in (1, 2):
        raise ValueError(f"Unexpected generated audio shape: {audio.shape}")
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak * 0.98
    sf.write(str(path), audio, sample_rate)

st.subheader("1. Upload your short-film scene (optional)")
video = st.file_uploader("Choose a video file", type=["mp4", "mov", "mkv", "webm"])
if video:
    st.video(video.getvalue())

st.subheader("2. Describe the scene")
left, right = st.columns(2)
with left:
    scene = st.text_area(
        "What happens in the scene?",
        "A cricket fan confidently supports Sunrisers Hyderabad (SRH) and challenges a friend who supports Kolkata Knight Riders (KKR), saying SRH will win.",
        height=130,
    )
    mood = st.selectbox("Intended emotion", ["Hope", "Triumph", "Suspense", "Happiness", "Sadness", "Romance", "Fear", "Peaceful"])
with right:
    pacing = st.selectbox("Scene pacing", ["Slow", "Moderate", "Fast"], index=1)
    duration = st.slider("Target soundtrack duration (seconds)", 5, 30, 10, 5)
    instruments = st.text_input("Preferred instruments", "piano, soft strings, cinematic percussion")

st.caption("For a first CPU test, choose 5 or 10 seconds. MusicGen may produce a slightly different duration.")

styles = [
    ("Track A — Emotional", "Emotional cinematic melody with expressive piano and warm strings"),
    ("Track B — Atmospheric", "Atmospheric cinematic ambient music with evolving soft textures"),
    ("Track C — Rhythmic", "Uplifting cinematic music with a steady rhythmic pulse and percussion"),
]

if "tracks" not in st.session_state:
    st.session_state.tracks = []

st.subheader("3. Generate three soundtrack alternatives")
if st.button("✨ Generate 3 AI soundtracks", type="primary"):
    if not scene.strip():
        st.error("Please describe your scene.")
    else:
        try:
            processor, model = load_model()
            st.session_state.tracks = []
            progress = st.progress(0)
            status = st.empty()
            # MusicGen uses about 50 audio tokens per second.
            max_new_tokens = max(128, int(duration * 50))

            for index, (label, style) in enumerate(styles, start=1):
                prompt = build_prompt(style, mood, pacing, scene, instruments or "cinematic orchestra")
                status.write(f"Generating {label} locally. Please wait...")
                inputs = processor(text=[prompt], padding=True, return_tensors="pt")
                with torch.inference_mode():
                    generated = model.generate(
                        **inputs,
                        do_sample=True,
                        guidance_scale=3,
                        max_new_tokens=max_new_tokens,
                    )
                sample_rate = model.config.audio_encoder.sampling_rate
                output_path = OUTPUT_DIR / f"track_{index}.wav"
                save_audio(generated[0], sample_rate, output_path)
                st.session_state.tracks.append({"label": label, "path": str(output_path), "prompt": prompt})
                progress.progress(index / len(styles))
                del inputs, generated
                gc.collect()

            status.success("All three soundtracks were generated!")
        except Exception as exc:
            st.error("Local music generation failed.")
            st.exception(exc)
            st.info("Check PyTorch/Transformers installation, internet access for the first model download, RAM, and free disk space.")

tracks = st.session_state.get("tracks", [])
if tracks:
    st.subheader("4. Listen to and compare the tracks")
    for track in tracks:
        st.markdown(f"**{track['label']}**")
        path = Path(track["path"])
        if path.exists():
            data = path.read_bytes()
            st.audio(data, format="audio/wav")
            st.download_button(
                f"Download {track['label']}", data=data,
                file_name=path.name, mime="audio/wav",
                key=f"download_{track['label']}"
            )
        with st.expander("View generation prompt"):
            st.write(track["prompt"])

    st.subheader("5. Evaluation")
    st.write("Listen to each version and score it yourself (1 = poor, 5 = excellent).")
    ratings = {}
    for i, track in enumerate(tracks, start=1):
        st.markdown(f"**{track['label']}**")
        c1, c2, c3 = st.columns(3)
        ratings[i] = {
            "emotion": c1.slider(f"Emotional match — Track {i}", 1, 5, 3, key=f"emotion_{i}"),
            "pacing": c2.slider(f"Pacing — Track {i}", 1, 5, 3, key=f"pacing_{i}"),
            "quality": c3.slider(f"Audio quality — Track {i}", 1, 5, 3, key=f"quality_{i}"),
        }

    timing_notes = st.text_area(
        "Timing notes",
        "00:00 — Music begins softly\n00:03 — Main melody enters\n00:07 — Music builds with the scene\nEnd — Music fades out"
    )
    if st.button("Save evaluation CSV"):
        csv_path = OUTPUT_DIR / "evaluation.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Track", "Emotional match (1-5)", "Pacing (1-5)", "Audio quality (1-5)", "Timing notes"])
            for i, track in enumerate(tracks, start=1):
                r = ratings[i]
                writer.writerow([track["label"], r["emotion"], r["pacing"], r["quality"], timing_notes])
        st.success("Evaluation saved.")
        st.download_button("Download evaluation.csv", csv_path.read_bytes(), file_name="evaluation.csv", mime="text/csv")

st.divider()
st.caption(
    "Academic prototype: generates alternative AI instrumental soundtracks and collects human ratings "
    "for emotional match, pacing, and audio quality. Scores are supplied by the evaluator; emotion is not automatically measured."
)
