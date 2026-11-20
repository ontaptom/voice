import base64
import json
import os
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Literal

import pymupdf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from google import genai
from google.genai import types
from pydantic import BaseModel

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
# MODEL = os.getenv("GEMINI_MODEL", "gemini-4-argon")

TTS_MODEL = os.getenv("GEMINI_TTS_MODEL", "gemini-3.8-flash-tts")
TTS_STYLE = os.getenv("GEMINI_TTS_STYLE", "casual, conversational, spontaneous")
BASE_DIR = Path(__file__).parent

app = FastAPI(title="Slide Presenter Demo")


class Turn(BaseModel):
    speaker: Literal["Speaker 1", "Speaker 2"]
    text: str


class SlideDialogue(BaseModel):
    turns: list[Turn]

# Simple in-memory deck state for local demo use.
deck: dict[str, object] = {
    "filename": None,
    "slides": [],  # list[bytes] of PNG images
}

_client: genai.Client | None = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client()
    return _client


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(BASE_DIR / "templates" / "tts.html")


@app.post("/api/upload")
async def upload_presentation(file: UploadFile = File(...)) -> dict:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    pdf_bytes = await file.read()
    try:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid PDF: {exc}") from exc

    slides: list[bytes] = []
    try:
        for page in doc:
            pix = page.get_pixmap(dpi=150)
            slides.append(pix.tobytes("png"))
    finally:
        doc.close()

    if not slides:
        raise HTTPException(status_code=400, detail="The PDF has no pages.")

    deck["filename"] = file.filename
    deck["slides"] = slides
    return {"filename": file.filename, "count": len(slides)}


@app.get("/api/slides/{index}")
async def get_slide(index: int) -> Response:
    slides: list[bytes] = deck["slides"]  # type: ignore[assignment]
    if index < 0 or index >= len(slides):
        raise HTTPException(status_code=404, detail="Slide not found.")
    return Response(content=slides[index], media_type="image/png")


def build_prompt(fillers: bool) -> str:
    if fillers:
        return (
            "You are two energetic co-presenters (\"Speaker 1\" and \"Speaker 2\") live on stage "
            "having a fast, dynamic conversation while presenting this slide to an audience. "
            "React actively to each other—chiming in with agreement, curiosity, or surprise—and "
            "speak directly as presenters (never comment on slide design, colors, backgrounds, "
            "or decorative graphics; only discuss informative charts or diagrams).\n\n"
            "Output format:\n"
            "Return JSON matching {\"turns\": [{\"speaker\": \"Speaker 1\" | \"Speaker 2\", \"text\": \"...\"}]}.\n"
            "- Each `text` must contain ONLY the spoken words (never include \"Speaker 1:\" or \"Speaker 2:\" prefixes).\n\n"
            "Conversation rules:\n"
            "- Generate 6 to 10 short turns alternating between \"Speaker 1\" and \"Speaker 2\".\n"
            "- Keep each turn very short: 1 punchy sentence (or fragment), max ~12 words.\n"
            "- Include natural spoken fillers where fitting (\"well\", \"actually\", \"you know\", \"I mean\", \"uh\", \"right?\").\n"
            "- Include a brief listener backchannel wrapped in pipes (`|...|`, 1 to 3 words) INSIDE the middle of nearly every turn "
            "(between phrases, never at the very start or very end of `text`), reflecting the other speaker actively reacting with "
            "agreement, curiosity, or surprise. Vary them instead of repeating the same one—for example: "
            "`|mm-hmm|`, `|yeah|`, `|right|`, `|totally|`, `|exactly|`, `|for sure|`, `|uh-huh|`, `|oh nice|`, "
            "`|oh really?|`, `|no way|`, `|wait really?|`, `|seriously?|`, `|whoa|`, `|true|`, `|interesting|`.\n"
            "- Use at most 1 or 2 inline vocal tags mid-sentence where natural (never right before trailing punctuation): `<short pause>`, `<breath>` (use `<laugh>` only if genuinely funny).\n"
            "- No markdown or parenthetical stage directions.\n\n"
            "Example JSON:\n"
            "{\n"
            "  \"turns\": [\n"
            "    {\"speaker\": \"Speaker 1\", \"text\": \"So today we're looking at |yeah| building real-time voice agents.\"},\n"
            "    {\"speaker\": \"Speaker 2\", \"text\": \"Wait, from prototype <short pause> all the way to |mm-hmm| production?\"},\n"
            "    {\"speaker\": \"Speaker 1\", \"text\": \"Exactly, and honestly |oh nice| the latency drop is huge.\"}\n"
            "  ]\n"
            "}"
        )

    return (
        "You are two co-presenters (\"Speaker 1\" and \"Speaker 2\") live on stage "
        "presenting this slide to an audience. "
        "Speak directly as presenters (never comment on slide design, colors, backgrounds, "
        "or decorative graphics; only discuss informative charts or diagrams).\n\n"
        "Output format:\n"
        "Return JSON matching {\"turns\": [{\"speaker\": \"Speaker 1\" | \"Speaker 2\", \"text\": \"...\"}]}.\n"
        "- Each `text` must contain ONLY the spoken words (never include \"Speaker 1:\" or \"Speaker 2:\" prefixes).\n\n"
        "Conversation rules:\n"
        "- Generate 6 to 10 short turns alternating between \"Speaker 1\" and \"Speaker 2\".\n"
        "- Keep each turn concise and clear: 1 sentence, max ~12 words.\n"
        "- Use clean, direct spoken language ONLY: NO filler words (\"uh\", \"um\", \"well\", \"like\", \"you know\"), "
        "NO pipe backchannels (`|...|`), NO angle-bracket vocal tags (`<...>`), NO markdown, and NO stage directions.\n\n"
        "Example JSON:\n"
        "{\n"
        "  \"turns\": [\n"
        "    {\"speaker\": \"Speaker 1\", \"text\": \"Today we are looking at building real-time voice agents.\"},\n"
        "    {\"speaker\": \"Speaker 2\", \"text\": \"We will cover everything from prototype to production.\"},\n"
        "    {\"speaker\": \"Speaker 1\", \"text\": \"The biggest improvement comes from cutting end-to-end latency.\"}\n"
        "  ]\n"
        "}"
    )


@app.post("/api/slides/{index}/describe")
async def describe_slide(index: int, fillers: bool = True) -> StreamingResponse:
    slides: list[bytes] = deck["slides"]  # type: ignore[assignment]
    if index < 0 or index >= len(slides):
        raise HTTPException(status_code=404, detail="Slide not found.")

    slide_bytes = slides[index]
    prompt = build_prompt(fillers)
    mode_label = "fillers+tags+style" if fillers else "plain (no fillers/tags/style)"

    async def event_stream() -> AsyncIterator[str]:
        def emit(event: dict) -> str:
            return json.dumps(event) + "\n"

        try:
            yield emit({"type": "llm_start", "model": MODEL, "mode": mode_label})
            client = get_client()
            start = time.perf_counter()
            response = await client.aio.models.generate_content(
                model=MODEL,
                contents=[
                    prompt,
                    types.Part.from_bytes(data=slide_bytes, mime_type="image/png"),
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=SlideDialogue,
                ),
            )
            elapsed = time.perf_counter() - start
            print(f"slide={index} mode={mode_label} model_used={MODEL} time={elapsed:.2f}s", flush=True)
            dialogue = response.parsed or SlideDialogue.model_validate_json(response.text or "{}")
            turns = [t.model_dump() for t in dialogue.turns]
            yield emit(
                {
                    "type": "llm_done",
                    "model": MODEL,
                    "elapsed": round(elapsed, 2),
                    "raw_text": response.text or "",
                    "turns": turns,
                }
            )

            yield emit({"type": "tts_start", "model": TTS_MODEL, "mode": mode_label})
            tts_start = time.perf_counter()
            audio_bytes, audio_mime = await synthesize_dialogue(
                dialogue.turns, include_style=fillers
            )
            tts_elapsed = time.perf_counter() - tts_start
            print(
                f"slide={index} mode={mode_label} tts_model={TTS_MODEL} time={tts_elapsed:.2f}s bytes={len(audio_bytes)}",
                flush=True,
            )
            yield emit(
                {
                    "type": "tts_done",
                    "model": TTS_MODEL,
                    "elapsed": round(tts_elapsed, 2),
                    "bytes": len(audio_bytes),
                    "audio_b64": base64.b64encode(audio_bytes).decode("ascii") if audio_bytes else None,
                    "audio_mime": audio_mime or "audio/wav",
                }
            )
        except Exception as exc:
            yield emit({"type": "error", "detail": f"Gemini error: {exc}"})

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


async def synthesize_dialogue(
    turns: list[Turn], include_style: bool = True
) -> tuple[bytes, str]:
    client = get_client()

    def build_annotation(speaker: str) -> dict[str, str]:
        ann = {"type": "speech_metadata", "speaker": speaker}
        if include_style:
            ann["style"] = TTS_STYLE
        return ann

    interaction = await client.aio.interactions.create(
        api_version="v1beta",
        timeout=60,
        model=TTS_MODEL,
        input=[
            {
                "type": "user_input",
                "content": [
                    {
                        "type": "text",
                        "text": turn.text,
                        "annotations": [build_annotation(turn.speaker)],
                    }
                    for turn in turns
                ],
            }
        ],
        response_format={"type": "audio"},
        generation_config={
            "temperature": 1,
            "top_p": 0.95,
            "speech_config": {
                "speakers": [
                    {"speaker": "Speaker 1", "voice": "Fola"},
                    {"speaker": "Speaker 2", "voice": "Rami"},
                ],
            },
        },
    )

    audio_data = bytearray()
    mime_type = ""
    for step in interaction.steps or []:
        if getattr(step, "type", None) != "model_output":
            continue
        for item in getattr(step, "content", None) or []:
            if getattr(item, "type", None) == "audio" and getattr(item, "data", None):
                raw = item.data
                audio_data.extend(
                    raw if isinstance(raw, (bytes, bytearray)) else base64.b64decode(raw)
                )
                mime_type = item.mime_type or mime_type
    return bytes(audio_data), mime_type
