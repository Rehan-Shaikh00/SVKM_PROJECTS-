"""Audio plumbing for telephony.

Twilio Media Streams (and most SIP/RTP bridges) exchange **8 kHz G.711 μ-law**,
20 ms frames. Cloud ASR/TTS services want 16/22/24 kHz PCM. This module owns
that conversion plus the energy-based voice activity detector used for
endpointing and barge-in.

Everything is pure numpy — no ffmpeg / no audioop dependency, so it works on
Python 3.13+ where `audioop` was removed.
"""

from __future__ import annotations

import base64
import math
from dataclasses import dataclass, field

import numpy as np

MULAW_BIAS = 0x84
MULAW_CLIP = 32635
FRAME_MS = 20
TELEPHONY_RATE = 8000

# --------------------------------------------------------------------------- #
# G.711 μ-law
# --------------------------------------------------------------------------- #

_MULAW_ENCODE_TABLE = np.zeros(65536, dtype=np.uint8)
_MULAW_DECODE_TABLE = np.zeros(256, dtype=np.int16)


def _build_tables() -> None:
    for i in range(256):
        sign = ~i & 0x80
        exponent = (~i >> 4) & 0x07
        mantissa = ~i & 0x0F
        value = ((mantissa << 3) + MULAW_BIAS) << exponent
        value -= MULAW_BIAS
        _MULAW_DECODE_TABLE[i] = -value if sign else value

    seg_end = np.array(
        [0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF, 0x3FFF, 0x7FFF], dtype=np.int32
    )
    for pcm in range(-32768, 32768):
        value = max(-MULAW_CLIP, min(MULAW_CLIP, pcm))
        sign = 0
        if value < 0:
            sign = 0x80
            value = -value
        value = min(value, MULAW_CLIP)
        seg = int(np.searchsorted(seg_end, value, side="left"))
        if seg >= 8:
            ulaw = sign | 0x7F
        else:
            if seg == 0:
                mantissa = value >> 4
            else:
                mantissa = (value + MULAW_BIAS) >> (seg + 3)
            ulaw = sign | (seg << 4) | (mantissa & 0x0F)
        _MULAW_ENCODE_TABLE[pcm & 0xFFFF] = ~ulaw & 0xFF


_build_tables()


def mulaw_to_pcm16(data: bytes) -> np.ndarray:
    """Decode G.711 μ-law bytes into signed 16-bit linear PCM samples."""
    if not data:
        return np.zeros(0, dtype=np.int16)
    idx = np.frombuffer(data, dtype=np.uint8).astype(np.int32)
    return _MULAW_DECODE_TABLE[idx].astype(np.int16)


def pcm16_to_mulaw(pcm: np.ndarray) -> bytes:
    """Encode signed 16-bit PCM into G.711 μ-law bytes."""
    if pcm.size == 0:
        return b""
    arr = np.ascontiguousarray(pcm, dtype=np.int16)
    return _MULAW_ENCODE_TABLE[arr.view(np.uint16).astype(np.int32)].tobytes()


def pcm16_to_float(pcm: np.ndarray) -> np.ndarray:
    return pcm.astype(np.float32) / 32768.0


def float_to_pcm16(audio: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16)


def b64_to_mulaw_pcm(b64: str) -> np.ndarray:
    return mulaw_to_pcm16(base64.b64decode(b64))


def pcm_to_b64_mulaw(pcm: np.ndarray) -> str:
    return base64.b64encode(pcm16_to_mulaw(pcm)).decode("ascii")


# --------------------------------------------------------------------------- #
# Resampling / framing
# --------------------------------------------------------------------------- #


def resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Linear-interpolation resampler.

    Speech-band quality is fine for 8k ↔ 16k/22.05k/24k conversion; for
    audiophile needs swap in soxr (see requirements comments).
    """
    if src_rate == dst_rate or audio.size == 0:
        return audio
    ratio = dst_rate / src_rate
    length = int(round(audio.size * ratio))
    if length <= 0:
        return np.zeros(0, dtype=audio.dtype)
    x_old = np.linspace(0.0, 1.0, num=audio.size, endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=length, endpoint=False)
    return np.interp(x_new, x_old, audio.astype(np.float32)).astype(audio.dtype)


def frame_count(samples: int, sample_rate: int, frame_ms: int = FRAME_MS) -> int:
    frame_size = int(sample_rate * frame_ms / 1000)
    return int(math.ceil(samples / frame_size)) if frame_size else 0


def iter_frames(pcm: np.ndarray, sample_rate: int, frame_ms: int = FRAME_MS):
    """Yield 20 ms PCM frames (the unit Twilio Media Streams expects)."""
    frame_size = int(sample_rate * frame_ms / 1000)
    if frame_size <= 0:
        return
    for start in range(0, pcm.size, frame_size):
        chunk = pcm[start : start + frame_size]
        if chunk.size < frame_size:  # pad the tail so timing stays constant
            chunk = np.concatenate(
                [chunk, np.zeros(frame_size - chunk.size, dtype=chunk.dtype)]
            )
        yield chunk


def silence_pcm(sample_rate: int, duration_ms: int, dtype=np.int16) -> np.ndarray:
    return np.zeros(int(sample_rate * duration_ms / 1000), dtype=dtype)


def dbfs(pcm: np.ndarray) -> float:
    """Rough level in dBFS for logging/debug."""
    if pcm.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(pcm.astype(np.float64) / 32768.0))))
    if rms <= 1e-9:
        return -120.0
    return 20.0 * math.log10(rms)


# --------------------------------------------------------------------------- #
# Voice activity detection
# --------------------------------------------------------------------------- #


@dataclass
class VADState:
    in_speech: bool = False
    speech_started_at: float | None = None
    silence_ms: float = 0.0
    speech_ms: float = 0.0
    noise_floor: float = 0.02
    history: list[float] = field(default_factory=list)


class EnergyVAD:
    """Adaptive energy VAD tuned for 8 kHz telephone audio.

    Telephone audio is noisy (line hiss, car backgrounds, fan noise), so the
    threshold adapts: the noise floor tracks a slow moving average of quiet
    frames and speech must exceed `floor + margin` for `min_speech_ms`.
    Hysteresis (different on/off thresholds) prevents flapping.
    """

    def __init__(
        self,
        sample_rate: int = TELEPHONY_RATE,
        frame_ms: int = FRAME_MS,
        on_threshold_db: float = -38.0,
        off_threshold_db: float = -46.0,
        min_speech_ms: int = 140,
        endpoint_silence_ms: int = 650,
        adaptive_noise_floor: bool = True,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.on_threshold = 10 ** (on_threshold_db / 20.0)
        self.off_threshold = 10 ** (off_threshold_db / 20.0)
        self.min_speech_frames = max(1, int(min_speech_ms / frame_ms))
        self.endpoint_frames = max(1, int(endpoint_silence_ms / frame_ms))
        self.adaptive = adaptive_noise_floor
        self.state = VADState()
        self._speech_run = 0
        self._silence_run = 0
        self._floor = 0.008
        self._frame_size = int(sample_rate * frame_ms / 1000)

    def reset(self) -> None:
        self.state = VADState()
        self._speech_run = 0
        self._silence_run = 0

    @property
    def endpoint_reached(self) -> bool:
        return self.state.in_speech and self._silence_run >= self.endpoint_frames

    def process(self, pcm: np.ndarray, now: float | None = None) -> dict[str, object]:
        """Feed one frame. Returns events: {'speech_started','speech_ended','rms'}."""
        frame = pcm.astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(np.square(frame)))) if frame.size else 0.0
        events: dict[str, object] = {"rms": rms, "speech": self.state.in_speech}

        # Zero-crossing rate helps reject hum/DTMF-ish tones that have high energy
        # but are not speech.
        zcr = 0.0
        if frame.size > 1:
            zcr = float(np.mean(np.abs(np.diff(np.sign(frame))) > 0))

        is_voice_like = rms > (self._floor * 2.2 if self.adaptive else self.off_threshold)
        speechy = rms >= self.on_threshold and 0.02 < zcr < 0.55

        if speechy or (is_voice_like and rms >= self.on_threshold * 0.8):
            self._speech_run += 1
            self._silence_run = 0
        else:
            self._silence_run += 1
            self._speech_run = 0
            if self.adaptive and not self.state.in_speech:
                # slowly track the noise floor upward/downward
                self._floor = 0.97 * self._floor + 0.03 * max(rms, 1e-5)

        if not self.state.in_speech and self._speech_run >= self.min_speech_frames:
            self.state.in_speech = True
            self.state.speech_started_at = now
            events["speech_started"] = True

        elif self.state.in_speech and self._silence_run >= self.endpoint_frames:
            self.state.in_speech = False
            self.state.speech_ms = self._speech_run * self.frame_ms
            events["speech_ended"] = True
            self._speech_run = 0
            self._silence_run = 0

        self.state.silence_ms = self._silence_run * self.frame_ms
        self.state.speech_ms = self._speech_run * self.frame_ms
        events["speech"] = self.state.in_speech
        events["silence_ms"] = self.state.silence_ms
        return events

    def process_mulaw(self, data: bytes, now: float | None = None) -> dict[str, object]:
        return self.process(mulaw_to_pcm16(data), now)


# --------------------------------------------------------------------------- #
# DTMF (last-resort fallback only)
# --------------------------------------------------------------------------- #

DTMF_FREQUENCIES = {
    "1": (697, 1209), "2": (697, 1336), "3": (697, 1477), "A": (697, 1633),
    "4": (770, 1209), "5": (770, 1336), "6": (770, 1477), "B": (770, 1633),
    "7": (852, 1209), "8": (852, 1336), "9": (852, 1477), "C": (852, 1633),
    "*": (941, 1209), "0": (941, 1336), "#": (941, 1477), "D": (941, 1633),
}


def dtmf_tone(digit: str, sample_rate: int = TELEPHONY_RATE, duration_ms: int = 120) -> np.ndarray:
    """Generate a DTMF tone (used by tests and the simulator, not in production)."""
    low, high = DTMF_FREQUENCIES.get(digit.upper(), (697, 1209))
    t = np.arange(int(sample_rate * duration_ms / 1000)) / sample_rate
    wave = 0.4 * np.sin(2 * np.pi * low * t) + 0.4 * np.sin(2 * np.pi * high * t)
    return (wave * 32767).astype(np.int16)


def wav_header(num_samples: int, sample_rate: int = TELEPHONY_RATE, channels: int = 1) -> bytes:
    """Minimal 16-bit PCM WAV header (used for storing debug captures)."""
    byte_rate = sample_rate * channels * 2
    data_size = num_samples * channels * 2
    return (
        b"RIFF" + (36 + data_size).to_bytes(4, "little") + b"WAVEfmt "
        + (16).to_bytes(4, "little")
        + (1).to_bytes(2, "little")
        + channels.to_bytes(2, "little")
        + sample_rate.to_bytes(4, "little")
        + byte_rate.to_bytes(4, "little")
        + (channels * 2).to_bytes(2, "little")
        + (16).to_bytes(2, "little")
        + b"data" + data_size.to_bytes(4, "little")
    )
