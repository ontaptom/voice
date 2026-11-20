import asyncio
import contextlib
import os
from pathlib import Path

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from google import genai
from google.genai import types

MODEL = os.getenv("GEMINI_TRANSLATE_MODEL", "gemini-3.5-live-translate-preview")
BASE_DIR = Path(__file__).parent

SUPPORTED_TARGETS: dict[str, str] = {
    "en": "English",
    "pl": "Polish",
    "de": "German",
    "fr": "French",
    "uk": "Ukrainian",
}

app = FastAPI(title="DevFest Warsaw — Live Translate Demo")


def get_client() -> genai.Client:
    return genai.Client(http_options={"api_version": "v1beta"})


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(BASE_DIR / "templates" / "livetranslate.html")


@app.websocket("/ws/translate")
async def ws_translate(
    ws: WebSocket,
    target: str = Query(default="en"),
    transcripts: bool = Query(default=True),
) -> None:
    await ws.accept()

    target_code = target.lower()
    if target_code not in SUPPORTED_TARGETS:
        await ws.send_json(
            {"type": "error", "detail": f"Unsupported target language: {target}"}
        )
        await ws.close()
        return

    config_kwargs: dict[str, object] = {
        "response_modalities": ["AUDIO"],
        "translation_config": types.TranslationConfig(
            target_language_code=target_code,
            echo_target_language=False,
        ),
    }
    if transcripts:
        config_kwargs["input_audio_transcription"] = types.AudioTranscriptionConfig()
        config_kwargs["output_audio_transcription"] = types.AudioTranscriptionConfig()

    config = types.LiveConnectConfig(**config_kwargs)  # type: ignore[arg-type]

    print(
        f"live_translate connect model={MODEL} target={target_code} transcripts={transcripts}",
        flush=True,
    )

    try:
        client = get_client()
        async with client.aio.live.connect(model=MODEL, config=config) as session:
            await ws.send_json(
                {
                    "type": "status",
                    "state": "connected",
                    "model": MODEL,
                    "target": target_code,
                    "target_name": SUPPORTED_TARGETS[target_code],
                    "transcripts": transcripts,
                }
            )

            async def browser_to_gemini() -> None:
                while True:
                    msg = await ws.receive()
                    msg_type = msg.get("type")
                    if msg_type == "websocket.disconnect":
                        break
                    if msg.get("bytes"):
                        await session.send_realtime_input(
                            audio=types.Blob(
                                data=msg["bytes"],
                                mime_type="audio/pcm;rate=16000",
                            )
                        )
                    elif msg.get("text") == "stop":
                        break

            async def gemini_to_browser() -> None:
                while True:
                    got_any = False
                    async for response in session.receive():
                        got_any = True
                        sc = response.server_content
                        if not sc:
                            continue

                        if transcripts and sc.input_transcription:
                            tr = sc.input_transcription
                            if tr.text or tr.language_code:
                                await ws.send_json(
                                    {
                                        "type": "input_transcript",
                                        "text": tr.text or "",
                                        "language_code": tr.language_code,
                                        "finished": bool(tr.finished),
                                    }
                                )

                        if transcripts and sc.output_transcription:
                            tr = sc.output_transcription
                            if tr.text or tr.language_code:
                                await ws.send_json(
                                    {
                                        "type": "output_transcript",
                                        "text": tr.text or "",
                                        "language_code": tr.language_code or target_code,
                                        "finished": bool(tr.finished),
                                    }
                                )

                        if sc.model_turn and sc.model_turn.parts:
                            for part in sc.model_turn.parts:
                                if part.inline_data and part.inline_data.data:
                                    await ws.send_bytes(part.inline_data.data)

                    if not got_any:
                        break

            in_task = asyncio.create_task(browser_to_gemini())
            out_task = asyncio.create_task(gemini_to_browser())
            done, pending = await asyncio.wait(
                {in_task, out_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
            for task in done:
                exc = task.exception()
                if exc and not isinstance(exc, WebSocketDisconnect):
                    raise exc

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        print(f"live_translate error: {exc}", flush=True)
        with contextlib.suppress(Exception):
            await ws.send_json({"type": "error", "detail": str(exc)})
    finally:
        print(f"live_translate disconnect target={target_code}", flush=True)
        with contextlib.suppress(Exception):
            await ws.close()
