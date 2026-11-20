import asyncio
import contextlib
import json
import os
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from google import genai
from google.genai import types

MODEL = os.getenv("GEMINI_TRANSCRIBE_LIVE_MODEL", "gemini-3.5-transcribe-live")
TRANSLATE_MODEL = os.getenv("GEMINI_TRANSLATE_TEXT_MODEL", "gemini-3.5-flash-lite")
BASE_DIR = Path(__file__).parent

app = FastAPI(title="DevFest Warsaw — Live Transcribe Demo")


def get_live_client() -> genai.Client:
    return genai.Client(http_options={"api_version": "v1beta"})


def get_text_client() -> genai.Client:
    return genai.Client()


def build_translation_prompt(text: str, target_language: str) -> str:
    return (
        f"Translate the following spoken transcript into {target_language}. "
        "Preserve formatting (line breaks, bullets) if present. "
        "Return ONLY the translated text with no quotes, labels, or commentary.\n\n"
        f"{text}"
    )


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(BASE_DIR / "templates" / "transcribe.html")


@app.websocket("/ws/transcribe")
async def ws_transcribe(ws: WebSocket) -> None:
    await ws.accept()

    try:
        init_raw = await ws.receive_text()
        init_msg = json.loads(init_raw)
    except (WebSocketDisconnect, json.JSONDecodeError) as exc:
        with contextlib.suppress(Exception):
            await ws.send_json({"type": "error", "detail": f"Invalid init message: {exc}"})
            await ws.close()
        return

    smart = bool(init_msg.get("smart", False))
    mode = "SMART" if smart else "VERBATIM"
    raw_vocab = init_msg.get("custom_vocabulary") or []
    custom_vocab = [str(term).strip() for term in raw_vocab if str(term).strip()][:1000]
    translate_enabled = bool(init_msg.get("translate", False))
    target_language = str(init_msg.get("target_language") or "English").strip() or "English"

    transcription_cfg_kwargs: dict[str, object] = {
        "language_codes": [],  # Auto-detect + code-switching
        "mode": mode,
    }
    if custom_vocab:
        transcription_cfg_kwargs["custom_vocabulary"] = custom_vocab

    config = types.LiveConnectConfig(
        response_modalities=["TEXT"],
        input_audio_transcription=types.AudioTranscriptionConfig(
            **transcription_cfg_kwargs  # type: ignore[arg-type]
        ),
    )

    print(
        f"live_transcribe connect model={MODEL} mode={mode} vocab_count={len(custom_vocab)} "
        f"translate={translate_enabled} target_language={target_language!r}",
        flush=True,
    )

    ws_lock = asyncio.Lock()

    async def send_json_safe(payload: dict) -> None:
        async with ws_lock:
            await ws.send_json(payload)

    generation_seq = 0
    latest_interim: tuple[int, str] | None = None
    interim_event = asyncio.Event()
    final_queue: asyncio.Queue[tuple[str, str, bool]] = asyncio.Queue()

    try:
        live_client = get_live_client()
        text_client = get_text_client() if translate_enabled else None

        async def interim_translate_worker() -> None:
            nonlocal latest_interim
            assert text_client is not None
            while True:
                await interim_event.wait()
                interim_event.clear()
                # Brief coalesce window to avoid flooding on rapid partial updates
                await asyncio.sleep(0.18)
                if latest_interim is None:
                    continue
                seq, text = latest_interim
                latest_interim = None
                if not text.strip() or seq != generation_seq:
                    continue
                try:
                    resp = await text_client.aio.models.generate_content(
                        model=TRANSLATE_MODEL,
                        contents=build_translation_prompt(text, target_language),
                        config=types.GenerateContentConfig(temperature=0.1),
                    )
                    translated = (resp.text or "").strip()
                    if translated and seq == generation_seq:
                        await send_json_safe(
                            {
                                "type": "translation_interim",
                                "text": translated,
                            }
                        )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    print(f"interim translate error: {exc}", flush=True)

                if latest_interim is not None:
                    interim_event.set()

        async def final_translate_worker() -> None:
            assert text_client is not None
            while True:
                item_mode, text, finished = await final_queue.get()
                # In SMART mode, drain superseded cumulative snapshots to stay fresh
                if item_mode == "SMART":
                    while not final_queue.empty():
                        item_mode, text, finished = final_queue.get_nowait()

                if not text.strip():
                    continue
                try:
                    stream = await text_client.aio.models.generate_content_stream(
                        model=TRANSLATE_MODEL,
                        contents=build_translation_prompt(text, target_language),
                        config=types.GenerateContentConfig(temperature=0.1),
                    )
                    full_chunks: list[str] = []
                    first_chunk = True
                    async for chunk in stream:
                        piece = chunk.text or ""
                        if not piece:
                            continue
                        full_chunks.append(piece)
                        await send_json_safe(
                            {
                                "type": "translation_final_delta",
                                "mode": item_mode,
                                "delta": piece,
                                "reset": first_chunk and item_mode == "SMART",
                                "new_segment": first_chunk and item_mode != "SMART",
                            }
                        )
                        first_chunk = False
                    await send_json_safe(
                        {
                            "type": "translation_final_done",
                            "mode": item_mode,
                            "text": "".join(full_chunks).strip(),
                            "finished": finished,
                        }
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    print(f"final translate error: {exc}", flush=True)
                    await send_json_safe(
                        {"type": "error", "detail": f"Translation error: {exc}"}
                    )

        async with live_client.aio.live.connect(model=MODEL, config=config) as session:
            await send_json_safe(
                {
                    "type": "status",
                    "state": "connected",
                    "model": MODEL,
                    "mode": mode,
                    "vocab_count": len(custom_vocab),
                    "translate": translate_enabled,
                    "target_language": target_language,
                    "translate_model": TRANSLATE_MODEL if translate_enabled else None,
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
                        with contextlib.suppress(Exception):
                            await session.send_realtime_input(audio_stream_end=True)
                        await asyncio.sleep(0.5)
                        break

            async def gemini_to_browser() -> None:
                nonlocal generation_seq, latest_interim
                while True:
                    got_any = False
                    async for response in session.receive():
                        got_any = True
                        sc = response.server_content
                        if not sc:
                            continue

                        if sc.interim_input_transcription:
                            tr = sc.interim_input_transcription
                            if tr.text is not None or tr.language_code:
                                interim_txt = tr.text or ""
                                await send_json_safe(
                                    {
                                        "type": "interim",
                                        "text": interim_txt,
                                        "language_code": tr.language_code,
                                    }
                                )
                                if translate_enabled and interim_txt.strip():
                                    latest_interim = (generation_seq, interim_txt)
                                    interim_event.set()

                        if sc.input_transcription:
                            tr = sc.input_transcription
                            if tr.text is not None or tr.language_code:
                                final_txt = tr.text or ""
                                await send_json_safe(
                                    {
                                        "type": "final",
                                        "mode": mode,
                                        "text": final_txt,
                                        "language_code": tr.language_code,
                                        "finished": bool(tr.finished),
                                    }
                                )
                                if translate_enabled:
                                    generation_seq += 1
                                    latest_interim = None
                                    if final_txt.strip():
                                        await final_queue.put(
                                            (mode, final_txt, bool(tr.finished))
                                        )

                    if not got_any:
                        break

            tasks: list[asyncio.Task[None]] = [
                asyncio.create_task(browser_to_gemini()),
                asyncio.create_task(gemini_to_browser()),
            ]
            bg_tasks: list[asyncio.Task[None]] = []
            if translate_enabled:
                bg_tasks.append(asyncio.create_task(interim_translate_worker()))
                bg_tasks.append(asyncio.create_task(final_translate_worker()))

            done, pending = await asyncio.wait(
                tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in list(pending) + bg_tasks:
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
        print(f"live_transcribe error: {exc}", flush=True)
        with contextlib.suppress(Exception):
            await send_json_safe({"type": "error", "detail": str(exc)})
    finally:
        print(f"live_transcribe disconnect mode={mode}", flush=True)
        with contextlib.suppress(Exception):
            await ws.close()
