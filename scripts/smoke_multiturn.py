"""Drive a multi-turn call and check the assistant carries context.

The brief asks for context across turns: a caller who says "what is the
eligibility for B.Tech Computer Engineering?" and then "and the fees?" must not
be asked which programme they mean. Also exercises a mid-call language switch
and a barge-in.
"""

import asyncio
import json
import time

import websockets

URL = "ws://127.0.0.1:8000/api/simulator/ws"

TURNS = [
    (2.0, {"type": "text", "text": "English"}),
    (3.0, {"type": "text", "text": "What is the eligibility for B.Tech Computer Engineering?"}),
    (3.0, {"type": "text", "text": "And what about the fees?"}),
    (3.0, {"type": "text", "text": "How many seats are there?"}),
    (3.0, {"type": "text", "text": "What about the pharmacy one?"}),
    # Naming a language outright is a choice, not a question: this must switch
    # the call, and must not be answered. It used to come back with the academic
    # calendar, because that record carries Marathi aliases.
    (3.0, {"type": "text", "text": "मराठी"}),
    # A caller who simply starts speaking another language needs two turns, since
    # one high-confidence detection is not enough to flip the whole call.
    (3.0, {"type": "text", "text": "सेमेस्टर कधी सुरू होईल?"}),
    (3.0, {"type": "text", "text": "त्याची पात्रता काय आहे?"}),
    # Barge-in is a control message, not a message type of its own.
    (1.0, {"type": "control", "name": "barge_in"}),
    (1.5, {"type": "text", "text": "When do classes start?"}),
    (3.0, {"type": "hangup"}),
]


async def main() -> int:
    started = time.time()
    utterances: list[str] = []
    events: list[dict] = []
    async with websockets.connect(URL, max_size=None) as ws:
        await ws.send(json.dumps({"type": "start", "from": "+919000000000",
                                  "to": "+912562350620", "mode": "text"}))

        async def reader() -> None:
            try:
                while True:
                    msg = json.loads(await ws.recv())
                    kind = msg.get("type")
                    events.append(msg)
                    if kind == "audio":
                        continue
                    text = msg.get("text") or ""
                    if kind == "assistant_text" and text:
                        utterances.append(text)
                    stamp = f"[{time.time() - started:6.2f}s]"
                    if kind in ("assistant_text", "caller_transcript", "state",
                                "language_switched", "language_detected", "transfer",
                                "call_ended", "error", "barge_in", "interrupted",
                                "control_ack"):
                        extra = "" if text else " " + json.dumps(
                            {k: v for k, v in msg.items() if k != "type"},
                            ensure_ascii=False)[:120]
                        print(f"{stamp} <- {kind:17s} {text[:118]}{extra}")
                    if kind in ("call_ended", "error"):
                        return
            except websockets.ConnectionClosed:
                return

        reader_task = asyncio.create_task(reader())
        for delay, message in TURNS:
            await asyncio.sleep(delay)
            if reader_task.done():
                break
            print(f"[{time.time() - started:6.2f}s] -> {message.get('type'):17s} "
                  f"{message.get('text', '')[:80]}")
            await ws.send(json.dumps(message))
        await asyncio.wait_for(reader_task, timeout=25)

    print(f"\nassistant utterances: {len(utterances)}")
    tally: dict[str, int] = {}
    for event in events:
        tally[event.get("type", "?")] = tally.get(event.get("type", "?"), 0) + 1
    print("event tally:", json.dumps(tally, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
