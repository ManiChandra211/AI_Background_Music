# AI Background Music for a Short Film

## Fresh installation in VS Code (Windows)

1. Extract this ZIP file.
2. Open the extracted `AI_Background_Music_Fresh_Project` folder in VS Code.
3. Open Terminal > New Terminal.
4. Create a virtual environment:

   ```powershell
   python -m venv .venv
   ```

5. Activate it:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, use Command Prompt and run:
   `\.venv\Scripts\activate.bat`

6. Install dependencies:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

7. Create a Hugging Face token at https://huggingface.co/settings/tokens.
   The token must have permission to use Inference Providers. Keep it private.
8. Run the app:

   ```powershell
   python -m streamlit run app.py
   ```

9. Open the local URL shown by Streamlit, usually http://localhost:8501.
10. Paste the token in the left sidebar, upload a short video, describe the scene,
    select emotion and pacing, then click Generate 3 AI soundtracks.

## What it does

- Generates three prompted soundtrack alternatives.
- Plays and downloads the generated audio.
- Optionally combines each track with the uploaded scene as an MP4.
- Collects human ratings for emotional match, pacing, and audio quality.
- Exports the evaluation as CSV in `outputs/`.

## Important

This application uses Hugging Face hosted inference. It requires internet, a valid token,
and a model that is available through an inference provider for your account. The default
model is `facebook/musicgen-small`; if the provider reports that the model is unavailable,
choose another compatible text-to-audio model available to your account.

The selected duration is included in the prompt; the hosted model may not follow it exactly.
The video export replaces the original video audio. Ratings are entered by the user and are
not automatic emotion-recognition results.
