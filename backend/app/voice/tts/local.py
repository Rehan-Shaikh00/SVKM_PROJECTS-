"""On-host TTS and the client-speech path.

`LocalTTS` shells out to espeak-ng or Piper if one of them is installed. It is a
development/offline fallback — telephone quality is poor, so production should
use a cloud neural voice. When no engine exists on the host we degrade to
`text_only`, which the browser simulator turns into Web Speech synthesis.

`ClientTTS` is the mode used by the simulator and by any client that renders
speech itself (web widget, mobile SDK): the backend sends text, the client speaks.
"""

from __future__ import annotations

import asyncio
import shutil
import struct
import time

import numpy as np

from ...config import settings
from ...i18n.languages import get_language, voice_for
from ..audio import pcm16_to_mulaw, resample
from .base import TTS, SynthesisResult, sanitize_for_speech

ESPEAK_LANG_MAP = {
    "en-IN": "en", "hi-IN": "hi", "raj-IN": "hi", "ta-IN": "ta", "bn-IN": "bn",
    "mr-IN": "mr", "gu-IN": "gu", "te-IN": "te", "kn-IN": "kn", "ml-IN": "ml",
    "pa-IN": "pa", "ur-IN": "ur", "or-IN": "or", "as-IN": "as",
}


def detect_local_engine() -> str | None:
    for binary in ("espeak-ng", "espeak", "piper"):
        if shutil.which(binary):
            return binary
    return None


class LocalTTS(TTS):
    name = "local"
    encodings = ("mulaw", "pcm16")

    def __init__(self, engine: str | None = None) -> None:
        super().__init__()
        self.engine = engine or detect_local_engine()
        self.text_only = self.engine is None

    async def synthesize(
        self, text: str, language: str = "en-IN", *, sample_rate: int = 8000,
        encoding: str = "mulaw", speaking_rate: float | None = None,
    ) -> SynthesisResult:
        clean = sanitize_for_speech(text)
        if not clean:
            return SynthesisResult(text="", provider=self.name, text_only=True, is_last=True)
        if not self.engine:
            return SynthesisResult(
                text=clean, provider=self.name, language=language, text_only=True,
                encoding="none", characters=len(clean),
            )
        started = time.time()
        pcm, rate = await asyncio.to_thread(self._run_engine, clean, language, speaking_rate)
        latency = self._timed(started)
        audio = resample(pcm, rate, sample_rate).astype(np.int16)
        return SynthesisResult(
            audio=pcm16_to_mulaw(audio) if encoding == "mulaw" else audio.tobytes(),
            encoding="mulaw" if encoding == "mulaw" else "pcm16",
            sample_rate=sample_rate,
            text=clean,
            voice=f"{self.engine}:{ESPEAK_LANG_MAP.get(language, 'en')}",
            provider=self.name,
            language=language,
            latency_ms=latency,
            characters=len(clean),
        )

    def _run_engine(
        self, text: str, language: str, speaking_rate: float | None
    ) -> tuple[np.ndarray, int]:  # pragma: no cover - depends on host binaries
        import subprocess
        import tempfile
        from pathlib import Path

        rate_multiplier = speaking_rate or settings.tts_speaking_rate
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            out_path = Path(handle.name)
        try:
            if self.engine in ("espeak-ng", "espeak"):
                lang = ESPEAK_LANG_MAP.get(language, voice_for(language, "local") or "en")
                cmd = [
                    self.engine, "-v", lang, "-s", str(int(155 * rate_multiplier)),
                    "-w", str(out_path), "--", text,
                ]
                subprocess.run(cmd, check=True, capture_output=True, timeout=25)
            else:  # piper
                cmd = ["piper", "--output_file", str(out_path)]
                # check=False on purpose: piper's non-zero exit is handled below
                # so the caller gets the engine's stderr in the error message.
                proc = subprocess.run(
                    cmd,
                    input=text.encode("utf-8"),
                    capture_output=True,
                    timeout=60,
                    check=False,
                )
                if proc.returncode != 0:
                    raise RuntimeError(proc.stderr.decode(errors="ignore")[:200])
            return _read_wav(out_path.read_bytes())
        finally:
            out_path.unlink(missing_ok=True)


class ClientTTS(TTS):
    """Backend produces no audio; the client renders speech itself."""

    name = "client"
    text_only = True
    encodings = ("none",)

    async def synthesize(
        self, text: str, language: str = "en-IN", *, sample_rate: int = 8000,
        encoding: str = "mulaw", speaking_rate: float | None = None,
    ) -> SynthesisResult:
        started = time.time()
        clean = sanitize_for_speech(text)
        return SynthesisResult(
            audio=b"",
            encoding="none",
            sample_rate=sample_rate,
            text=clean,
            voice=f"client:{get_language(language).native_name}",
            provider=self.name,
            language=language,
            text_only=True,
            latency_ms=self._timed(started),
            characters=len(clean),
        )


def _read_wav(data: bytes) -> tuple[np.ndarray, int]:
    """Minimal RIFF/WAV reader (16-bit PCM)."""
    if len(data) < 44 or data[:4] != b"RIFF":
        return np.zeros(0, dtype=np.int16), 8000
    channels = struct.unpack("<H", data[22:24])[0] or 1
    rate = struct.unpack("<I", data[24:28])[0] or 22050
    bits = struct.unpack("<H", data[34:36])[0] or 16
    idx = data.find(b"data")
    if idx < 0:
        return np.zeros(0, dtype=np.int16), rate
    size = struct.unpack("<I", data[idx + 4 : idx + 8])[0]
    payload = data[idx + 8 : idx + 8 + size]
    if bits == 16:
        pcm = np.frombuffer(payload, dtype="<i2")
    else:
        pcm = ((np.frombuffer(payload, dtype=np.uint8).astype(np.int16) - 128) * 256).astype(
            np.int16
        )
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1).astype(np.int16)
    return pcm, rate
