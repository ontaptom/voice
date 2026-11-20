# DevFest Warsaw 2026 - Voice Demos

Local Python demos built with `uv` (Python 3.13+) and the Gemini API.

## Quickstart

1. Install dependencies:
   ```bash
   uv sync
   ```

2. Set your Gemini API key:
   ```bash
   export GEMINI_API_KEY="your-api-key"
   ```

3. Run one of the demos:

   - **Multi-Voice Slide Presenter:**
     ```bash
     uv run uvicorn tts:app --reload
     ```
     Open `http://127.0.0.1:8000`, attach a PDF presentation, navigate slides, and click **Present Slide**.

   - **Live Translate:**
     ```bash
     uv run uvicorn livetranslate:app --reload
     ```
     Open `http://127.0.0.1:8000`, pick a target language, and click **Start Live Translate** for real-time speech-to-speech translation.

   - **Live Transcribe:**
     ```bash
     uv run uvicorn transcribe:app --reload
     ```
     Open `http://127.0.0.1:8000`, optionally enable Smart Transcription or custom vocabulary, and click **Start Transcribing**.

   - **Live Async Tools:**
     ```bash
     uv run uvicorn livetools:app --reload
     ```
     Open `http://127.0.0.1:8000`, pick a live model and blocking/non-blocking tool mode, and click **Start Live Session**.

## Optional Environment Variables

- `GEMINI_MODEL` - slide dialogue model (default: `gemini-3.5-flash-lite`)
- `GEMINI_TTS_MODEL` - multi-speaker TTS model (default: `gemini-3.8-flash-tts`)
- `GEMINI_TTS_STYLE` - per-turn TTS style prompt (default: `casual, conversational, spontaneous`)

## Project Layout

- `tts.py` + `templates/tts.html` - FastAPI Multi-Voice Slide Presenter web app
- `livetranslate.py` + `templates/livetranslate.html` - FastAPI Live Translation web app
- `transcribe.py` + `templates/transcribe.html` - FastAPI Live Transcription web app
- `livetools.py` + `templates/livetools.html` - FastAPI Live Voice Agent async tool-calling web app

## Author & Disclaimer

**Tomek Porozynski**

This repository contains "quick and dirty" demos built to explore and showcase Gemini's audio stack models. These apps are **not production-ready**, have **not** been hardened or tested for security, authentication, or public exposure, and are intended strictly for local demos rather than being published or deployed online.

## License

Licensed under the [Apache License 2.0](LICENSE).
