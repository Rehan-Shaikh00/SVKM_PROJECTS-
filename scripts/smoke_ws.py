"""End-to-end WebSocket smoke test against the running simulator.

Drives a real `CallSession` exactly like the browser phone does: start the call,
listen to the greeting, pick a language, ask questions, barge in, then hang up.
Prints every server event so the conversation flow can be eyeballed.
"""

import asyncio
import json
import sys
import time

import websockets

URL = "ws://127.0.0.1:8000/api/simulator/ws"

# (delay before sending, message)
SCRIPT = [
    (2.0, {"type": "text", "text": "Hindi"}),
    (2.5, {"type": "text", "text": "बीटेक की फीस कितनी है"}),
    (3.0, {"type": "text", "text": "एमबीबीएस में एडमिशन कैसे लें"}),
    (3.0, {"type": "text", "text": "hostel mandatory hai kya"}),
    (3.0, {"type": "text", "text": "what documents do I need"}),
    (3.0, {"type": "text", "text": "मुझे किसी व्यक्ति से बात करनी है"}),
    (3.0, {"type": "hangup"}),
]


async def main() -> int:
    started = time.time()
    events: list[dict] = []
    async with websockets.connect(URL, max_size=None) as ws:
        await ws.send(json.dumps({"type": "start", "from": "+919000000000",
                                  "to": "+911800120102", "mode": "text"}))

        async def reader() -> None:
            try:
                while True:
                    raw = await ws.recv()
                    msg = json.loads(raw)
                    kind = msg.get("type")
                    if kind == "audio":
                        events.append({"type": "audio", "bytes": len(msg.get("payload", ""))})
                        continue
                    events.append(msg)
                    text = msg.get("text") or msg.get("message") or ""
                    print(f"[{time.time()-started:6.2f}s] <- {kind:16s} {str(text)[:110]}")
                    if not text and kind not in ("audio",):
                        extra = {k: v for k, v in msg.items() if k != "type"}
                        print(f"                 {json.dumps(extra, ensure_ascii=False)[:150]}")
                    if kind in ("call_ended", "hangup", "error"):
                        return
            except websockets.ConnectionClosed:
                return

        reader_task = asyncio.create_task(reader())

        for delay, message in SCRIPT:
            await asyncio.sleep(delay)
            if reader_task.done():
                break  # server already ended the call
            print(f"[{time.time()-started:6.2f}s] -> {message.get('type'):16s} "
                  f"{str(message.get('text', ''))[:70]}")
            try:
                await ws.send(json.dumps(message))
            except websockets.ConnectionClosed:
                break
            if message.get("type") == "hangup":
                break

        try:
            await asyncio.wait_for(reader_task, timeout=12)
        except TimeoutError:
            reader_task.cancel()

    kinds = {}
    for e in events:
        kinds[e.get("type")] = kinds.get(e.get("type"), 0) + 1
    print("\nevent tally:", json.dumps(kinds, ensure_ascii=False))
    spoken = [e for e in events if e.get("type") in ("speak", "tts", "say", "assistant_text")]
    print("assistant utterances:", len(spoken))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
