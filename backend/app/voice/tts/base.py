"""TTS interface + sentence chunking.

Latency is the product here: a voice assistant that pauses >1.5 s feels broken.
So synthesis is **sentence-streamed** — the first sentence is spoken while the
later ones are still being synthesised. `split_for_speech` keeps chunks short
and never splits inside a number, a course code or an abbreviation.
"""

from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass

from ..audio import pcm16_to_mulaw, resample

# sentence terminators incl. Devanagari danda (।) and Dravidian/Arabic punctuation
_SENTENCE_END = re.compile(r"(?<=[.!?।॥؟])\s+|(?<=[。！？])\s*")
_PROTECTED = re.compile(r"\b(?:[A-Z]\.){2,}|\b\d+(?:[.,]\d+)*\b")


@dataclass
class SynthesisResult:
    """One chunk of synthesised speech."""

    audio: bytes = b""
    encoding: str = "mulaw"          # mulaw | pcm16 | mp3 | ogg
    sample_rate: int = 8000
    text: str = ""
    voice: str = ""
    provider: str = ""
    language: str = "en-IN"
    #: no audio produced — the client (browser/widget) should speak `text`
    text_only: bool = False
    latency_ms: float = 0.0
    characters: int = 0
    is_last: bool = True

    def as_pcm16(self, target_rate: int = 8000):
        """Return numpy int16 PCM at `target_rate`, decoding if necessary."""
        import numpy as np

        from ..audio import mulaw_to_pcm16

        if self.text_only or not self.audio:
            return np.zeros(0, dtype=np.int16)
        if self.encoding == "mulaw":
            pcm = mulaw_to_pcm16(self.audio)
            rate = self.sample_rate or 8000
        else:  # pcm16
            pcm = np.frombuffer(self.audio, dtype=np.int16)
            rate = self.sample_rate
        if rate != target_rate:
            pcm = resample(pcm, rate, target_rate)
        return pcm.astype(np.int16)

    def as_mulaw(self, target_rate: int = 8000) -> bytes:
        pcm = self.as_pcm16(target_rate)
        return pcm16_to_mulaw(pcm)


def split_for_speech(text: str, max_chars: int = 180, min_chars: int = 24) -> list[str]:
    """Split into speakable chunks: sentence boundaries first, then clauses."""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    sentences = [s.strip() for s in _SENTENCE_END.split(text) if s and s.strip()]
    chunks: list[str] = []
    buffer = ""
    for sentence in sentences:
        if len(sentence) > max_chars:
            if buffer:
                chunks.append(buffer)
                buffer = ""
            chunks.extend(_split_long(sentence, max_chars))
            continue
        candidate = f"{buffer} {sentence}".strip() if buffer else sentence
        if len(candidate) <= max_chars:
            buffer = candidate
        else:
            if buffer:
                chunks.append(buffer)
            buffer = sentence
    if buffer:
        chunks.append(buffer)

    # merge tiny trailing fragments so the voice does not chop
    merged: list[str] = []
    for chunk in chunks:
        if merged and len(chunk) < min_chars and len(merged[-1]) + len(chunk) < max_chars + 40:
            merged[-1] = f"{merged[-1]} {chunk}"
        else:
            merged.append(chunk)
    return [c.strip() for c in merged if c.strip()]


def _split_long(sentence: str, max_chars: int) -> list[str]:
    """Break a long sentence at commas / conjunctions, protecting numbers."""
    # Split *before* a conjunction, using a lookahead, so the conjunction itself
    # survives. A consuming split silently deleted every "and"/"और" from long
    # answers ("Class 10 marksheet and certificate" became "…marksheet
    # certificate"). "per" is not a conjunction and must never be a split point,
    # or "per year" / "per annum" lose their unit.
    parts = re.split(
        r"(?<=[,;:])\s+|\s+(?=(?:and|or|but|और|या|लेकिन|तथा)\s)",
        sentence,
    )
    out: list[str] = []
    buffer = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        candidate = f"{buffer} {part}" if buffer else part
        if len(candidate) <= max_chars:
            buffer = candidate
            continue
        # Flush the buffer *once*. Slicing `candidate` after flushing it used to
        # re-emit the buffer's text, so callers spoke the same clause twice.
        if buffer:
            out.append(buffer)
            buffer = ""
        # Only the overflowing part itself may need hard slicing.
        while len(part) > max_chars:
            cut = part.rfind(" ", 0, max_chars)
            if cut <= 0:
                cut = max_chars
            out.append(part[:cut].strip())
            part = part[cut:].strip()
        buffer = part
    if buffer:
        out.append(buffer)
    return [o for o in out if o]


def sanitize_for_speech(text: str) -> str:
    """Make text voice-friendly before synthesis.

    * strip markdown, bullets, URLs, emoji
    * expand currency and common abbreviations the way Indians say them
    * collapse whitespace (TTS pauses on newlines)
    """
    if not text:
        return ""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"(?<!\w)\*(.+?)\*(?!\w)", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"__([A-Z_][A-Z0-9_]*)__", lambda m: m.group(1).title(), text)
    text = re.sub(r"#{1,6}\s*", "", text)
    text = re.sub(r"^\s*[-*•]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]", "", text)
    text = text.replace("|", ", ").replace("—", ", ").replace("–", ", ")

    # currency: ₹1,50,000 -> "one lakh fifty thousand rupees" is provider specific,
    # so keep numerals but make the unit speakable
    text = re.sub(r"₹\s*", "rupees ", text)
    text = re.sub(r"\bINR\s*", "rupees ", text)
    text = re.sub(r"\bRs\.?\s*", "rupees ", text)
    text = re.sub(r"\brupees (\d[\d,]*)\b", r"rupees \1", text)
    text = re.sub(r"\bLPA\b", "lakh per annum", text)
    text = re.sub(r"\bL\.?P\.?A\.?\b", "lakh per annum", text)
    text = re.sub(r"\bPh\.?D\.?\b", "P H D", text)
    text = re.sub(r"\bB\.Tech\b", "B Tech", text)
    text = re.sub(r"\bM\.Tech\b", "M Tech", text)
    text = re.sub(r"\bB\.Sc\b", "B Sc", text)
    text = re.sub(r"\bM\.Sc\b", "M Sc", text)
    text = re.sub(r"\bB\.A\b", "B A", text)
    text = re.sub(r"\bM\.A\b", "M A", text)
    text = re.sub(r"\bB\.Com\b", "B Com", text)
    text = re.sub(r"\bM\.Com\b", "M Com", text)
    text = re.sub(r"\bB\.Pharm\b", "B Pharm", text)
    text = re.sub(r"\bM\.Pharm\b", "M Pharm", text)
    text = re.sub(r"\bB\.A\. LL\.?B\.?\b", "B A LL B", text)
    text = re.sub(r"\bB\.B\.A\.? LL\.?B\.?\b", "B B A LL B", text)
    text = re.sub(r"\bLL\.?B\.?\b", "LL B", text)
    text = re.sub(r"\bLL\.?M\.?\b", "LL M", text)
    text = re.sub(r"\bB\.P\.T\.?\b", "B P T", text)
    text = re.sub(r"\bB\.O\.T\.?\b", "B O T", text)
    text = re.sub(r"\bB\.Des\b", "B Des", text)
    text = re.sub(r"\bM\.Des\b", "M Des", text)
    text = re.sub(r"\bB\.Arch\b", "B Arch", text)
    text = re.sub(r"\bM\.Arch\b", "M Arch", text)
    text = re.sub(r"\bPharm\.D\b", "Pharm D", text)
    text = re.sub(r"\bMBBS\b", "M B B S", text)
    text = re.sub(r"\bBDS\b", "B D S", text)
    text = re.sub(r"\bMDS\b", "M D S", text)
    text = re.sub(r"\bMBA\b", "M B A", text)
    text = re.sub(r"\bBBA\b", "B B A", text)
    text = re.sub(r"\bMCA\b", "M C A", text)
    text = re.sub(r"\bBCA\b", "B C A", text)
    text = re.sub(r"\bBHMCT\b", "B H M C T", text)
    text = re.sub(r"\bCSE\b", "C S E", text)
    text = re.sub(r"\bECE\b", "E C E", text)
    text = re.sub(r"\bAI\b(?!\w)", "A I", text)
    text = re.sub(r"\bML\b(?!\w)", "M L", text)
    text = re.sub(r"\bNEET\b", "N E E T", text)
    text = re.sub(r"\bJEE\b", "J E E", text)
    text = re.sub(r"\bGATE\b", "G A T E", text)
    text = re.sub(r"\bCAT\b", "C A T", text)
    text = re.sub(r"\bNPAT\b", "N P A T", text)
    text = re.sub(r"\bNMAT\b", "N M A T", text)
    text = re.sub(r"\bCLAT\b", "C L A T", text)
    text = re.sub(r"\bLSAT\b", "L S A T", text)
    text = re.sub(r"\bGPAT\b", "G P A T", text)
    text = re.sub(r"\bSVKM\b", "S V K M", text)
    # "NMIMS" is said as letters; left to a TTS engine it becomes "nimms".
    text = re.sub(r"\bNMIMS\b", "N M I M S", text)
    text = re.sub(r"\bUGC\b", "U G C", text)
    text = re.sub(r"\bAICTE\b", "A I C T E", text)
    text = re.sub(r"\bNAAC\b", "N A A C", text)
    text = re.sub(r"\bNMC\b", "N M C", text)
    text = re.sub(r"\bPCI\b", "P C I", text)
    text = re.sub(r"\bBCI\b", "B C I", text)
    text = re.sub(r"\bCOA\b", "C O A", text)
    text = re.sub(r"\bSC\b", "S C", text)
    text = re.sub(r"\bST\b", "S T", text)
    text = re.sub(r"\bOBC\b", "O B C", text)
    text = re.sub(r"\bSMS\b", "S M S", text)
    text = re.sub(r"\bWhatsApp\b", "WhatsApp", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class TTS(ABC):
    """Text-to-speech engine."""

    name = "base"
    #: encodings this engine can emit natively
    encodings: tuple[str, ...] = ("mulaw",)
    #: languages covered (empty = any)
    supported_languages: tuple[str, ...] = ()
    text_only: bool = False

    def __init__(self) -> None:
        self.last_latency_ms = 0.0
        self.synthesis_count = 0

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        language: str = "en-IN",
        *,
        sample_rate: int = 8000,
        encoding: str = "mulaw",
        speaking_rate: float | None = None,
    ) -> SynthesisResult:
        """Synthesize one chunk."""

    async def synthesize_stream(
        self,
        text: str,
        language: str = "en-IN",
        *,
        sample_rate: int = 8000,
        encoding: str = "mulaw",
        speaking_rate: float | None = None,
        max_chunk_chars: int = 180,
    ) -> AsyncIterator[SynthesisResult]:
        """Yield chunks as soon as each is ready (first audio ≈ first sentence)."""
        chunks = split_for_speech(text, max_chars=max_chunk_chars)
        if not chunks:
            return
        for index, chunk in enumerate(chunks):
            result = await self.synthesize(
                chunk,
                language,
                sample_rate=sample_rate,
                encoding=encoding,
                speaking_rate=speaking_rate,
            )
            result.is_last = index == len(chunks) - 1
            yield result

    def _timed(self, started: float) -> float:
        latency = (time.time() - started) * 1000
        self.last_latency_ms = latency
        self.synthesis_count += 1
        return latency


class NullTTS(TTS):
    """Emits no audio. Used when the client speaks the text itself."""

    name = "null"
    text_only = True

    async def synthesize(
        self, text: str, language: str = "en-IN", *, sample_rate: int = 8000,
        encoding: str = "mulaw", speaking_rate: float | None = None,
    ) -> SynthesisResult:
        started = time.time()
        return SynthesisResult(
            audio=b"",
            encoding="none",
            sample_rate=sample_rate,
            text=text,
            provider=self.name,
            language=language,
            text_only=True,
            latency_ms=self._timed(started),
            characters=len(text),
        )
