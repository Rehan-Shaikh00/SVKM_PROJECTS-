"""Guardrails: PII handling, voice-style enforcement and grounding checks.

Two of these are load-bearing for a *phone* assistant:

* `scrub_for_voice` — models happily emit markdown, URLs and six-item lists. On a
  phone that is unusable, so the answer is rewritten/truncated before TTS.
* `numeric_grounding_check` — an answer that mentions a number which appears
  nowhere in the retrieved context is, by definition, invented. We catch it and
  escalate rather than reading a made-up fee to an applicant's family.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field

from ..config import settings

logger = logging.getLogger("nims.guardrails")

# --------------------------------------------------------------------------- #
# PII
# --------------------------------------------------------------------------- #

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]{2,}")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?91[-\s]?)?(?:[6-9]\d{4}[-\s]?\d{5})(?!\d)")
LANDLINE_RE = re.compile(r"(?<!\d)(?:0\d{2,4}[-\s]?)?\d{6,8}(?!\d)")
AADHAAR_RE = re.compile(r"(?<!\d)\d{4}[\s-]?\d{4}[\s-]?\d{4}(?!\d)")
PAN_RE = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")
CARD_RE = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")


def _mask(value: str, keep: int = 2) -> str:
    value = value.strip()
    if len(value) <= keep:
        return "*" * len(value)
    return "*" * (len(value) - keep) + value[-keep:]


def redact_pii(text: str) -> tuple[str, dict[str, list[str]]]:
    """Mask PII before anything is written to the database or logs."""
    if not settings.redact_pii or not text:
        return text, {}
    found: dict[str, list[str]] = {}
    result = text

    for label, pattern in (
        ("email", EMAIL_RE),
        ("url", URL_RE),
        ("pan", PAN_RE),
    ):
        matches = pattern.findall(result)
        if matches:
            found[label] = [hash_secret(m) for m in matches]
            # Bind `label` as a default argument: a bare closure would resolve it
            # at call time, so every redaction in the loop would be tagged with
            # the *last* label ("[pan:…]" for emails and URLs alike).
            result = pattern.sub(
                lambda m, _label=label: f"[{_label}:{_mask(m.group(0), 2)}]", result
            )

    def _phone_sub(match: re.Match[str]) -> str:
        found.setdefault("phone", []).append(hash_secret(match.group(0)))
        return f"[phone:{_mask(match.group(0), 3)}]"

    result = PHONE_RE.sub(_phone_sub, result)

    for label, pattern in (("aadhaar", AADHAAR_RE), ("card", CARD_RE)):
        def _sub(match: re.Match[str], label: str = label) -> str:
            digits = re.sub(r"\D", "", match.group(0))
            if label == "aadhaar" and len(digits) != 12:
                return match.group(0)
            if label == "card" and not (13 <= len(digits) <= 19):
                return match.group(0)
            found.setdefault(label, []).append(hash_secret(match.group(0)))
            return f"[{label}:{_mask(digits, 2)}]"

        result = pattern.sub(_sub, result)

    return result, found


def hash_secret(value: str) -> str:
    """Keyed pseudonym for a caller's phone number, Aadhaar or card digits.

    Keyed rather than a plain digest so the same number maps to the same
    pseudonym across calls -- follow-up SMS dedupe depends on that -- without
    the number itself ever being stored.

    The key is APP_SECRET, which is why a deployment left on the published
    placeholder has not pseudonymised anything. Indian mobile numbers are a
    small enough space to enumerate, so anyone who has read this repository can
    compute the digest of a number they suspect and confirm it against a stored
    `hash:` value. `main._enforce_credential_hygiene` refuses to boot outside
    development while the key is the placeholder.
    """
    digest = hashlib.sha256(f"{settings.app_secret}:{value.strip()}".encode())
    return digest.hexdigest()[:16]


# --------------------------------------------------------------------------- #
# voice style
# --------------------------------------------------------------------------- #

MAX_SPOKEN_WORDS = 60
MAX_SPOKEN_CHARS = 380

BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+", re.MULTILINE)
MARKDOWN_RE = re.compile(r"(?:\*\*|__|~~|`{1,3}|#{1,6}\s|>\s)")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?।?])\s+")


@dataclass
class VoiceScrubResult:
    text: str
    truncated: bool = False
    stripped_markdown: bool = False
    removed_items: int = 0
    word_count: int = 0
    warnings: list[str] = field(default_factory=list)


def scrub_for_voice(
    text: str,
    *,
    max_words: int = MAX_SPOKEN_WORDS,
    max_chars: int = MAX_SPOKEN_CHARS,
    language: str = "en-IN",
) -> VoiceScrubResult:
    """Force the answer into something that can actually be spoken."""
    original = text or ""
    result = original
    stripped = False
    removed_items = 0

    if MARKDOWN_RE.search(result):
        result = MARKDOWN_RE.sub("", result)
        stripped = True
    result = URL_RE.sub("", result)
    result = re.sub(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]", "", result)

    bullets = BULLET_RE.findall(result)
    if bullets:
        removed_items = max(0, len(bullets) - 3)
        lines = [BULLET_RE.sub("", line).strip() for line in result.splitlines() if line.strip()]
        if len(lines) > 3:
            head, rest = lines[:3], lines[3:]
            result = ". ".join(head)
            result += f". And {len(rest)} more items — I can send you the full list."
        else:
            result = ". ".join(lines)
        stripped = True

    result = re.sub(r"\s+", " ", result).strip()

    truncated = False
    words = result.split()
    if len(words) > max_words or len(result) > max_chars:
        sentences = SENTENCE_SPLIT_RE.split(result)
        kept: list[str] = []
        length = 0
        count = 0
        for sentence in sentences:
            if length + len(sentence) > max_chars or count + len(sentence.split()) > max_words:
                break
            kept.append(sentence)
            length += len(sentence) + 1
            count += len(sentence.split())
        result = " ".join(kept).strip() or (
            " ".join(words[:max_words]).rsplit(" ", 1)[0] + "."
        )
        truncated = True

    if len(result) > max_chars:
        result = result[:max_chars].rsplit(" ", 1)[0] + "."
        truncated = True

    if not result:
        result = original.strip()[:max_chars]

    warnings: list[str] = []
    if removed_items:
        warnings.append(f"removed {removed_items} list items for voice")
    if truncated:
        warnings.append("answer truncated for voice length")
    return VoiceScrubResult(
        text=result,
        truncated=truncated,
        stripped_markdown=stripped,
        removed_items=removed_items,
        word_count=len(result.split()),
        warnings=warnings,
    )


# --------------------------------------------------------------------------- #
# grounding
# --------------------------------------------------------------------------- #

NUMBER_RE = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")
INDIC_NUMBER_WORDS = (
    "lakh", "lakhs", "lac", "crore", "crores", "thousand", "hundred", "हज़ार",
    "हजार", "लाख", "करोड़", "सौ", "हज़ार", "हजार", "लाख", "कोटी",
)

#: numbers that are never a hallucinated fee
SAFE_NUMBERS = {
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "100",
    "2024", "2025", "2026", "2027", "10th", "12th",
}


#: Scale words a caller or the TTS formatter may put after a numeral.
SCALE_WORDS: dict[str, float] = {
    "crore": 10_000_000, "crores": 10_000_000, "cr": 10_000_000,
    "lakh": 100_000, "lakhs": 100_000, "lac": 100_000, "lacs": 100_000,
    "million": 1_000_000, "thousand": 1_000, "hundred": 100,
    "करोड़": 10_000_000, "करोड": 10_000_000, "कोटी": 10_000_000,
    "लाख": 100_000, "लक्ष": 100_000, "लक्ख": 100_000,
    "हज़ार": 1_000, "हजार": 1_000, "हजारा": 1_000, "सहस्र": 1_000,
    "सौ": 100, "हज़ारों": 1_000,
}
_SCALE_ALTERNATION = "|".join(
    re.escape(k) for k in sorted(SCALE_WORDS, key=len, reverse=True)
)
SCALED_NUMBER_RE = re.compile(
    rf"(\d[\d,]*(?:\.\d+)?)\s*({_SCALE_ALTERNATION})(?![\w])",
    re.IGNORECASE,
)
#: How much whitespace/punctuation may sit between "1 लाख" and "40 हज़ार"
#: before they stop being one amount.
_MAX_GROUP_GAP = 4


def _scaled_amounts(text: str) -> tuple[set[str], list[tuple[int, int]]]:
    """Resolve Indian scale phrases such as "1 लाख 40 हज़ार" -> "140000".

    Returns the values plus the character spans they consumed, so `_numbers` can
    avoid also emitting the bare components. Without this, the Hindi fee template
    ("1 लाख 40 हज़ार") reports an ungrounded "40" and the call escalates for no
    reason.
    """
    values: set[str] = set()
    spans: list[tuple[int, int]] = []
    group: list[tuple[int, int, float]] = []

    def flush() -> None:
        if not group:
            return
        total = sum(item[2] for item in group)
        if total:
            values.add(f"{total:.0f}")
        spans.append((group[0][0], group[-1][1]))

    for match in SCALED_NUMBER_RE.finditer(text):
        scale = SCALE_WORDS.get(match.group(2).lower())
        if scale is None:
            continue
        try:
            amount = float(match.group(1).replace(",", "")) * scale
        except ValueError:
            continue
        if group and match.start() - group[-1][1] > _MAX_GROUP_GAP:
            flush()
            group = []
        group.append((match.start(), match.end(), amount))
    flush()
    return values, spans


def _numbers(text: str) -> set[str]:
    text = text or ""
    out: set[str] = set()
    scaled, scaled_spans = _scaled_amounts(text)
    out |= scaled

    def _consumed(start: int, end: int) -> bool:
        return any(s <= start and end <= e for s, e in scaled_spans)

    for match in NUMBER_RE.finditer(text):
        if _consumed(match.start(), match.end()):
            continue
        compact = match.group(0).replace(",", "")
        out.add(compact)
        # allow "1.5 lakh" style to match "150000" in the context
        try:
            value = float(compact)
        except ValueError:
            continue
        out.add(f"{value:.0f}")
        out.add(f"{value * 100000:.0f}")   # lakh
        out.add(f"{value * 100_00_000:.0f}")  # crore
    return out


def numeric_grounding_check(answer: str, context: str) -> dict[str, object]:
    """Flag numbers spoken in the answer that appear nowhere in the context."""
    answer_numbers = _numbers(answer)
    context_numbers = _numbers(context)
    # also allow numbers spelled out in the context ("one lakh fifty thousand")
    suspicious = {
        n for n in answer_numbers
        if n not in context_numbers and n not in SAFE_NUMBERS and len(n) > 1
    }
    # percentages/years are common and low-risk; keep them out of the escalation path
    suspicious = {n for n in suspicious if not re.fullmatch(r"20\d{2}", n)}
    return {
        "ok": not suspicious,
        "answer_numbers": sorted(answer_numbers),
        "context_numbers": sorted(context_numbers),
        "ungrounded_numbers": sorted(suspicious),
    }


# --------------------------------------------------------------------------- #
# topic safety
# --------------------------------------------------------------------------- #

OFF_TOPIC_PATTERNS = (
    r"\b(stock|share price|bitcoin|crypto|betting|lottery)\b",
    r"\b(weather|cricket score|match result|movie|song|joke|poem)\b",
    r"\b(write (me )?(an )?essay|homework|assignment)\b",
    r"\b(bomb|weapon|drug|drugs|smuggle)\b",
    r"\b(sex|porn|nude)\b",
    r"\b(suicide|kill myself|self harm)\b",
    r"\b(vote|election|political party|religion conversion)\b",
)
OFF_TOPIC_RE = re.compile("|".join(OFF_TOPIC_PATTERNS), re.IGNORECASE)

SENSITIVE_PATTERNS = (
    # Harm and grievance. Match verb/adjective forms as well as nouns: a caller
    # says "my daughter was harassed on campus", not "harassment", and that must
    # reach a human being rather than an AI answer.
    r"\b(ragging|ragged|harass(?:ed|es|ing|ment)?|molest(?:ed|ing|ation)?"
    r"|assault(?:ed|ing)?|abuse[ds]?|abusing|discriminat\w*|caste slur|beaten"
    r"|threat(?:s|ened|ening)?|blackmail(?:ed|ing)?|extort(?:ed|ing|ion)?)\b",
    # Legal and financial disputes an AI must never adjudicate.
    r"\b(refund dispute|fee refund not|cheating|fraud|scam|court case|legal notice"
    r"|consumer court|lawyer|advocate|rti|complain(?:t|ts|ted|ting)?)\b",
    # Mental-health distress: hand off to a human immediately.
    r"\b(suicid\w*|self harm|depress\w*|panic|hopeless)\b",
    # Hindi / Rajasthani equivalents (substring match on purpose, so inflected
    # forms such as "शिकायत करना" are still caught).
    r"(शिकायत|उत्पीड़न|छेड़छाड़|बलात्कार|धमकी|धोखाधड़ी|रैगिंग|आत्महत्या|कानूनी|वकील|अदालत)",
)
SENSITIVE_RE = re.compile("|".join(SENSITIVE_PATTERNS), re.IGNORECASE)

UNIVERSITY_TOPIC_RE = re.compile(
    r"\b(course|courses|admission|admissions|fee|fees|eligib|scholarship|hostel|"
    r"placement|campus|exam|entrance|neet|jee|gate|cat|next|nimsee|document|"
    r"b\.?tech|m\.?tech|mba|mbbs|bds|b\.?pharm|pharm\.?d|mca|bca|b\.?sc|m\.?sc|"
    r"b\.?a\b|m\.?a\b|b\.?com|phd|nursing|physiotherapy|law|design|architecture|"
    r"hotel|journalism|agriculture|seat|seats|cutoff|merit|counsel(l)?ing|faculty|"
    r"library|lab|transport|bus|uniform|result|syllabus|semester)\b|"
    r"कोर्स|एडमिशन|फीस|पात्रता|छात्रवृत्ति|हॉस्टल|प्लेसमेंट|परिक्षा|परीक्षा|प्रवेश|आवेदन|"
    r"सीट|दस्तावेज|कोचिंग|कैंपस|विश्वविद्यालय|यूनिवर्सिटी",
    re.IGNORECASE,
)


def classify_topic(text: str) -> tuple[str, float]:
    """Return ('off_topic' | 'sensitive' | 'on_topic', confidence)."""
    if not text or not text.strip():
        return "on_topic", 0.0
    if SENSITIVE_RE.search(text):
        return "sensitive", 0.9
    if OFF_TOPIC_RE.search(text):
        return "off_topic", 0.85
    if UNIVERSITY_TOPIC_RE.search(text):
        return "on_topic", 0.7
    return "unknown", 0.3


def is_off_topic(text: str) -> bool:
    return classify_topic(text)[0] == "off_topic"


def is_sensitive(text: str) -> bool:
    return classify_topic(text)[0] == "sensitive"


def looks_like_pii_request(text: str) -> bool:
    return bool(re.search(r"\b(aadhaar|pan card|otp|password|card number|cvv)\b", text, re.IGNORECASE))
