import asyncio
import base64
import contextlib
import json
import os
import time
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from google import genai
from google.genai import types

DEFAULT_LIVE_MODEL = os.getenv("GEMINI_LIVE_MODEL", "gemini-3.8-live")
SUPPORTED_LIVE_MODELS = (
    "gemini-3.8-live",
    "gemini-2.5-flash-native-audio-preview-12-2025",
)
IMAGE_MODEL = os.getenv("GEMINI_IMAGE_MODEL", "gemini-3.1-flash-lite-image")
BASE_DIR = Path(__file__).parent

VALID_SCHEDULING = {
    "WHEN_IDLE": types.FunctionResponseScheduling.WHEN_IDLE,
    "INTERRUPT": types.FunctionResponseScheduling.INTERRUPT,
    "SILENT": types.FunctionResponseScheduling.SILENT,
}

SYSTEM_INSTRUCTION = """\
You are a witty, energetic AI voice art-studio assistant demoing live on stage at DevFest Warsaw 2026.
Keep spoken responses concise, natural, and conversational (1-3 sentences unless asked for more).

You have two tools:
1. generate_image(prompt, title): Generates an image and displays it on screen.
   Call this whenever the user asks you to create, draw, paint, or generate an image.
2. ask_art_critic(title, artist_note): Sends the artwork to a human art critic on screen for a 1-5 rating and feedback.
   Call this whenever the user asks for the art critic's opinion, rating, or review.

When tools run asynchronously (NON_BLOCKING):
- After starting a tool, briefly let the user know it's underway and happily keep chatting while it runs.
- When a tool result arrives later, react naturally to the generated artwork or the human critic's rating and feedback.
"""

app = FastAPI(title="DevFest Warsaw — Live Async Tools Demo")


def get_live_client() -> genai.Client:
    return genai.Client(http_options={"api_version": "v1beta"})


def get_image_client() -> genai.Client:
    return genai.Client()


def build_tools(non_blocking: bool) -> list[types.Tool]:
    behavior = (
        types.Behavior.NON_BLOCKING if non_blocking else types.Behavior.BLOCKING
    )
    return [
        types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name="generate_image",
                    description=(
                        "Generate an image from a visual prompt using an image generation model "
                        "and display it on screen."
                    ),
                    behavior=behavior,
                    parameters_json_schema={
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "Detailed visual prompt describing the image to generate.",
                            },
                            "title": {
                                "type": "string",
                                "description": "Short 2-5 word title for the artwork.",
                            },
                        },
                        "required": ["prompt", "title"],
                    },
                ),
                types.FunctionDeclaration(
                    name="ask_art_critic",
                    description=(
                        "Send an artwork to the human art critic on screen and wait for "
                        "their 1-5 star rating and feedback."
                    ),
                    behavior=behavior,
                    parameters_json_schema={
                        "type": "object",
                        "properties": {
                            "title": {
                                "type": "string",
                                "description": "Title of the artwork to critique.",
                            },
                            "artist_note": {
                                "type": "string",
                                "description": "Short 1-sentence pitch or note from you to the critic.",
                            },
                        },
                        "required": ["title", "artist_note"],
                    },
                ),
            ]
        )
    ]


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(BASE_DIR / "templates" / "livetools.html")


@app.websocket("/ws/live")
async def ws_live(ws: WebSocket) -> None:
    await ws.accept()

    try:
        init_raw = await ws.receive_text()
        init_msg = json.loads(init_raw)
    except (WebSocketDisconnect, json.JSONDecodeError) as exc:
        with contextlib.suppress(Exception):
            await ws.send_json({"type": "error", "detail": f"Invalid init message: {exc}"})
            await ws.close()
        return

    requested_model = str(init_msg.get("model") or DEFAULT_LIVE_MODEL).strip()
    live_model = (
        requested_model
        if requested_model in SUPPORTED_LIVE_MODELS
        else DEFAULT_LIVE_MODEL
    )
    supports_scheduling = not live_model.startswith("gemini-2.5-")

    behavior_str = str(init_msg.get("behavior", "NON_BLOCKING")).upper()
    non_blocking = behavior_str != "BLOCKING"
    behavior_label = "NON_BLOCKING" if non_blocking else "BLOCKING"

    default_scheduling = str(init_msg.get("scheduling", "WHEN_IDLE")).upper()
    if default_scheduling not in VALID_SCHEDULING:
        default_scheduling = "WHEN_IDLE"

    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=SYSTEM_INSTRUCTION,
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
        tools=build_tools(non_blocking),
    )

    print(
        f"live_tools connect live_model={live_model} image_model={IMAGE_MODEL} "
        f"behavior={behavior_label} scheduling={default_scheduling if supports_scheduling else 'omitted'}",
        flush=True,
    )

    ws_lock = asyncio.Lock()
    session_lock = asyncio.Lock()

    async def send_json_safe(payload: dict) -> None:
        async with ws_lock:
            await ws.send_json(payload)

    async def send_bytes_safe(data: bytes) -> None:
        async with ws_lock:
            await ws.send_bytes(data)

    in_flight_image_tasks: dict[str, asyncio.Task[None]] = {}
    pending_critic_calls: dict[str, dict[str, str]] = {}

    try:
        live_client = get_live_client()
        image_client = get_image_client()

        async with live_client.aio.live.connect(model=live_model, config=config) as session:
            await send_json_safe(
                {
                    "type": "status",
                    "state": "connected",
                    "live_model": live_model,
                    "image_model": IMAGE_MODEL,
                    "behavior": behavior_label,
                    "scheduling": default_scheduling if supports_scheduling else "default (omitted)",
                }
            )

            async def send_tool_response(
                call_id: str,
                name: str,
                response_data: dict,
                scheduling_override: str | None = None,
            ) -> str | None:
                sched_name: str | None = None
                fr_kwargs: dict[str, object] = {
                    "id": call_id,
                    "name": name,
                    "response": response_data,
                }
                if non_blocking and supports_scheduling:
                    chosen = (scheduling_override or default_scheduling).upper()
                    if chosen not in VALID_SCHEDULING:
                        chosen = "WHEN_IDLE"
                    sched_name = chosen
                    fr_kwargs["scheduling"] = VALID_SCHEDULING[chosen]
                elif non_blocking:
                    sched_name = "DEFAULT"

                fr = types.FunctionResponse(**fr_kwargs)  # type: ignore[arg-type]
                async with session_lock:
                    await session.send_tool_response(function_responses=[fr])
                return sched_name

            async def run_generate_image(call_id: str, prompt: str, title: str) -> None:
                started = time.perf_counter()
                try:
                    resp = await image_client.aio.models.generate_content(
                        model=IMAGE_MODEL,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_modalities=["IMAGE"],
                            image_config=types.ImageConfig(
                                aspect_ratio="1:1",
                                image_size="1K",
                            ),
                        ),
                    )
                    elapsed = round(time.perf_counter() - started, 2)

                    image_bytes: bytes | None = None
                    mime_type = "image/png"
                    for candidate in resp.candidates or []:
                        if not candidate.content or not candidate.content.parts:
                            continue
                        for part in candidate.content.parts:
                            if part.inline_data and part.inline_data.data:
                                image_bytes = part.inline_data.data
                                mime_type = part.inline_data.mime_type or "image/png"
                                break
                        if image_bytes:
                            break

                    if not image_bytes:
                        raise RuntimeError("Image model returned no image bytes.")

                    b64 = base64.b64encode(image_bytes).decode("ascii")
                    await send_json_safe(
                        {
                            "type": "image_ready",
                            "id": call_id,
                            "title": title,
                            "prompt": prompt,
                            "elapsed_s": elapsed,
                            "data_url": f"data:{mime_type};base64,{b64}",
                        }
                    )

                    sched_used = await send_tool_response(
                        call_id=call_id,
                        name="generate_image",
                        response_data={
                            "status": "generated",
                            "title": title,
                            "prompt": prompt,
                            "generation_time_seconds": elapsed,
                            "displayed_on_screen": True,
                        },
                    )
                    await send_json_safe(
                        {
                            "type": "tool_response_sent",
                            "id": call_id,
                            "name": "generate_image",
                            "scheduling": sched_used,
                            "summary": f"generated '{title}' in {elapsed}s",
                        }
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    elapsed = round(time.perf_counter() - started, 2)
                    print(f"generate_image error ({call_id}): {exc}", flush=True)
                    await send_json_safe(
                        {
                            "type": "image_error",
                            "id": call_id,
                            "detail": str(exc),
                        }
                    )
                    with contextlib.suppress(Exception):
                        sched_used = await send_tool_response(
                            call_id=call_id,
                            name="generate_image",
                            response_data={
                                "status": "error",
                                "error": str(exc),
                                "elapsed_seconds": elapsed,
                            },
                        )
                        await send_json_safe(
                            {
                                "type": "tool_response_sent",
                                "id": call_id,
                                "name": "generate_image",
                                "scheduling": sched_used,
                                "summary": f"error: {exc}",
                            }
                        )
                finally:
                    in_flight_image_tasks.pop(call_id, None)

            async def browser_to_gemini() -> None:
                nonlocal default_scheduling
                while True:
                    msg = await ws.receive()
                    msg_type = msg.get("type")
                    if msg_type == "websocket.disconnect":
                        break
                    if msg.get("bytes"):
                        async with session_lock:
                            await session.send_realtime_input(
                                audio=types.Blob(
                                    data=msg["bytes"],
                                    mime_type="audio/pcm;rate=16000",
                                )
                            )
                    elif msg.get("text"):
                        raw = msg["text"]
                        if raw == "stop":
                            break
                        try:
                            data = json.loads(raw)
                        except json.JSONDecodeError:
                            continue

                        evt_type = data.get("type")
                        if evt_type == "stop":
                            break
                        elif evt_type == "set_scheduling":
                            candidate = str(data.get("scheduling", "")).upper()
                            if candidate in VALID_SCHEDULING:
                                default_scheduling = candidate
                        elif evt_type == "critic_submit":
                            call_id = str(data.get("id", ""))
                            pending = pending_critic_calls.pop(call_id, None)
                            if not pending:
                                continue
                            rating = int(data.get("rating", 3))
                            rating = max(1, min(5, rating))
                            feedback = str(data.get("feedback", "")).strip() or "No extra comment."
                            sched_override = data.get("scheduling")
                            sched_used = await send_tool_response(
                                call_id=call_id,
                                name="ask_art_critic",
                                response_data={
                                    "title": pending["title"],
                                    "rating": rating,
                                    "max_rating": 5,
                                    "feedback": feedback,
                                },
                                scheduling_override=str(sched_override) if sched_override else None,
                            )
                            await send_json_safe(
                                {
                                    "type": "tool_response_sent",
                                    "id": call_id,
                                    "name": "ask_art_critic",
                                    "scheduling": sched_used,
                                    "summary": f"{rating}/5 — {feedback}",
                                }
                            )

            async def gemini_to_browser() -> None:
                while True:
                    got_any = False
                    async for response in session.receive():
                        got_any = True

                        if response.tool_call and response.tool_call.function_calls:
                            for fc in response.tool_call.function_calls:
                                call_id = fc.id or f"call-{int(time.time() * 1000)}"
                                name = fc.name or ""
                                args = dict(fc.args or {})
                                await send_json_safe(
                                    {
                                        "type": "tool_call",
                                        "id": call_id,
                                        "name": name,
                                        "args": args,
                                    }
                                )

                                if name == "generate_image":
                                    prompt = str(args.get("prompt") or "").strip() or "Abstract art"
                                    title = str(args.get("title") or "Untitled").strip() or "Untitled"
                                    task = asyncio.create_task(
                                        run_generate_image(call_id, prompt, title)
                                    )
                                    in_flight_image_tasks[call_id] = task
                                elif name == "ask_art_critic":
                                    title = str(args.get("title") or "Latest Artwork").strip()
                                    artist_note = str(args.get("artist_note") or "").strip()
                                    pending_critic_calls[call_id] = {
                                        "title": title,
                                        "artist_note": artist_note,
                                    }
                                    await send_json_safe(
                                        {
                                            "type": "critic_request",
                                            "id": call_id,
                                            "title": title,
                                            "artist_note": artist_note,
                                        }
                                    )
                                else:
                                    await send_tool_response(
                                        call_id=call_id,
                                        name=name,
                                        response_data={"error": f"Unknown tool: {name}"},
                                    )

                        if response.tool_call_cancellation and response.tool_call_cancellation.ids:
                            cancelled_ids = list(response.tool_call_cancellation.ids)
                            for cid in cancelled_ids:
                                task = in_flight_image_tasks.pop(cid, None)
                                if task and not task.done():
                                    task.cancel()
                                pending_critic_calls.pop(cid, None)
                            await send_json_safe(
                                {
                                    "type": "tool_cancelled",
                                    "ids": cancelled_ids,
                                }
                            )

                        sc = response.server_content
                        if not sc:
                            continue

                        if sc.interrupted:
                            await send_json_safe({"type": "interrupted"})

                        if sc.input_transcription and sc.input_transcription.text:
                            await send_json_safe(
                                {
                                    "type": "input_transcript",
                                    "text": sc.input_transcription.text,
                                    "finished": bool(sc.input_transcription.finished),
                                }
                            )

                        if sc.output_transcription and sc.output_transcription.text:
                            await send_json_safe(
                                {
                                    "type": "output_transcript",
                                    "text": sc.output_transcription.text,
                                    "finished": bool(sc.output_transcription.finished),
                                }
                            )

                        if sc.model_turn and sc.model_turn.parts:
                            for part in sc.model_turn.parts:
                                if part.inline_data and part.inline_data.data:
                                    await send_bytes_safe(part.inline_data.data)

                    if not got_any:
                        break

            in_task = asyncio.create_task(browser_to_gemini())
            out_task = asyncio.create_task(gemini_to_browser())
            done, pending = await asyncio.wait(
                {in_task, out_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in list(pending) + list(in_flight_image_tasks.values()):
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
        print(f"live_tools error: {exc}", flush=True)
        with contextlib.suppress(Exception):
            await send_json_safe({"type": "error", "detail": str(exc)})
    finally:
        print(f"live_tools disconnect behavior={behavior_label}", flush=True)
        with contextlib.suppress(Exception):
            await ws.close()
