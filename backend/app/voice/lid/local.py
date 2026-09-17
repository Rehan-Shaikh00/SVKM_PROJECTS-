"""Lexical + script-based language identification.

This is the **local** LID engine: it needs no credentials and runs in-process in
under a millisecond. It works on the transcript of the caller's language
declaration and is also used to *confirm* what an acoustic LID service guessed.

Strategy, in order of confidence:
1. The caller literally named a language ("Hindi", "हिंदी", "मारवाड़ी") → ~0.99
2. Non-Latin script present → script decides the language family (~0.9),
   markers disambiguate within it (Devanagari: Hindi / Marathi / Rajasthani,
   Bengali script: Bengali / Assamese)
3. Latin script → English vs romanised-Hindi (Hinglish) vs romanised-Marathi vs
   romanised-Rajasthani, scored by weighted function-word markers

Acoustic LID (Azure/Deepgram/Google) stays the primary path in production; this
module is the fallback, the confirmer, and the thing that makes the demo run
with zero API keys.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from ...i18n.languages import (
    HINGLISH_MARATHI_MARKERS,
    HINGLISH_MARKERS,
    HINGLISH_RAJASTHANI_MARKERS,
    LANGUAGES,
    SCRIPT_BLOCKS,
    get_language,
)

# --------------------------------------------------------------------------- #
# Explicit language names — what a caller actually says in the first turn
# --------------------------------------------------------------------------- #

NAME_TO_CODE: dict[str, str] = {
    # English
    "english": "en-IN", "eng": "en-IN", "angrezi": "en-IN", "angrejee": "en-IN",
    "अंग्रेजी": "en-IN", "अंग्रेज़ी": "en-IN", "انگریزی": "en-IN", "ஆங்கிலம்": "en-IN",
    "ইংরেজি": "en-IN", "ఇంగ్లీష్": "en-IN", "ಇಂಗ್ಲಿಷ್": "en-IN", "ഇംഗ്ലീഷ്": "en-IN",
    "અંગ્રેજી": "en-IN",
    # Hindi
    "hindi": "hi-IN", "hindustani": "hi-IN", "hindee": "hi-IN",
    "हिंदी": "hi-IN", "हिन्दी": "hi-IN", "ہندی": "hi-IN", "ஹிந்தி": "hi-IN",
    "হিন্দি": "hi-IN", "హిందీ": "hi-IN", "ಹಿಂದಿ": "hi-IN", "ഹിന്ദി": "hi-IN",
    "હિન્દી": "hi-IN", "हिंदी में": "hi-IN",
    # Rajasthani / Marwari
    "rajasthani": "raj-IN", "rajsthani": "raj-IN", "marwari": "raj-IN",
    "marwadi": "raj-IN", "marvari": "raj-IN", "rajasthani bhasha": "raj-IN",
    "राजस्थानी": "raj-IN", "मारवाड़ी": "raj-IN", "माड़वाड़ी": "raj-IN",
    "राजस्थानी भाषा": "raj-IN",
    # Tamil
    "tamil": "ta-IN", "tamizh": "ta-IN", "தமிழ்": "ta-IN", "தமிழ": "ta-IN",
    # Bengali
    "bengali": "bn-IN", "bangla": "bn-IN", "bangala": "bn-IN", "বাংলা": "bn-IN",
    "বঙ্গ": "bn-IN",
    # Marathi
    "marathi": "mr-IN", "मराठी": "mr-IN", "मराठी भाषा": "mr-IN",
    # Gujarati
    "gujarati": "gu-IN", "ગુજરાતી": "gu-IN", "गुजराती": "gu-IN",
    # Telugu
    "telugu": "te-IN", "తెలుగు": "te-IN", "तेलुगु": "te-IN",
    # Kannada
    "kannada": "kn-IN", "kanada": "kn-IN", "ಕನ್ನಡ": "kn-IN",
    # Malayalam
    "malayalam": "ml-IN", "മലയാളം": "ml-IN",
    # Punjabi
    "punjabi": "pa-IN", "panjabi": "pa-IN", "ਪੰਜਾਬੀ": "pa-IN", "पंजाबी": "pa-IN",
    # Urdu
    "urdu": "ur-IN", "اردو": "ur-IN", "उर्दू": "ur-IN",
    # Odia
    "odia": "or-IN", "oriya": "or-IN", "ଓଡ଼ିଆ": "or-IN", "ओडिया": "or-IN",
    # Assamese
    "assamese": "as-IN", "অসমীয়া": "as-IN",
}

_WORD_RE = re.compile(r"[\wऀ-ॿঀ-৿਀-੿઀-૿ୀ-୿஀-௿ఀ-౿ಀ-೿ഀ-ൿ؀-ۿ]+", re.UNICODE)

#: weighted markers — function words beat content words for language ID
DEVANAGARI_MARKERS: dict[str, dict[str, float]] = {
    "hi-IN": {
        "है": 3.0, "हैं": 3.0, "और": 2.0, "का": 2.0, "की": 2.0, "के": 2.0,
        "मुझे": 3.0, "मैं": 2.5, "आप": 2.0, "क्या": 3.0, "को": 1.5, "में": 1.5,
        "करना": 2.5, "हूँ": 3.0, "हुन": 2.5, "बताइए": 3.0, "कितनी": 3.0,
        "कितना": 3.0, "चाहिए": 3.0, "लिए": 2.0, "यह": 1.5, "वह": 1.5, "नहीं": 2.5,
        "फीस": 1.0, "एडमिशन": 1.0, "पढ़ाई": 2.0, "बताओ": 2.5, "कैसे": 3.0,
        # Question words, auxiliaries and the -एगा future: the words a caller
        # actually puts in a sentence. Without them a Hindi question that happens
        # to avoid "है" or "की" scored nothing above the bare script vote.
        "कब": 3.0, "कहाँ": 3.0, "कहां": 3.0, "कौनसा": 3.0, "कौनसी": 3.0,
        "कौनसे": 3.0, "कौन": 2.5, "लगेगा": 3.0, "लगेगी": 3.0, "लगता": 2.5,
        "होगा": 3.0, "होगी": 3.0, "हुआ": 2.5, "हुई": 2.5, "मिलता": 3.0,
        "मिलती": 3.0, "मिलेगा": 3.0, "मिलेगी": 3.0, "बताएं": 3.0, "बताये": 2.5,
        "इसका": 3.0, "इसकी": 3.0, "उसका": 2.5, "लेकिन": 3.0, "क्योंकि": 3.0,
        "छात्र": 2.5, "विश्वविद्यालय": 3.0, "पाठ्यक्रम": 3.0, "तक": 2.0,
        "बाद": 2.0, "पहले": 2.5, "साक्षात्कार": 3.0, "दीजिए": 3.0, "देना": 2.5,
        "भरें": 2.5, "देखें": 2.5, "करें": 2.5, "करेंगे": 3.0, "रहा": 2.0,
        "रही": 2.0, "गया": 2.0, "गई": 2.0, "जाएगा": 3.0, "भी": 1.5,
    },
    "mr-IN": {
        "आहे": 3.5, "आहेस": 3.5, "आहोत": 3.5, "मी": 2.5, "माझा": 3.0, "माझी": 3.0,
        "तुमचा": 3.0, "तुमची": 3.0, "आपला": 2.5, "काय": 3.0, "किती": 3.0,
        "करावे": 3.0, "म्हणून": 3.5, "आणि": 2.5, "नाही": 2.5, "पाहिजे": 3.5,
        "शिकायचे": 3.0, "ठिकाणी": 3.0, "साठी": 2.5,
        # Marathi shares the Devanagari script with Hindi, so the script vote is
        # worth nothing on its own: a Marathi sentence with none of the words
        # below tied Hindi at 6.0 and lost on candidate order. These are the
        # Marathi-only question words, auxiliaries, the -ईल future and the
        # oblique inflections (च्या, मध्ये) that Hindi does not use.
        "कधी": 3.5, "कुठे": 3.5, "कुठं": 3.5, "कोणता": 3.5, "कोणती": 3.5,
        "कोणते": 3.5, "कोणत्या": 3.5, "कसा": 3.5, "कशी": 3.5, "कसे": 3.5,
        "लागते": 3.5, "लागतील": 3.5, "लागेल": 3.5, "लागणार": 3.5,
        "सुरू": 3.0, "होईल": 3.5, "होणार": 3.5, "होते": 3.0, "झाले": 3.0,
        "मिळते": 3.5, "मिळेल": 3.5, "मिळणार": 3.5, "सांगा": 3.5, "सांगितले": 3.0,
        "त्याची": 3.5, "त्याचे": 3.5, "त्यांची": 3.5, "याची": 3.0, "हे": 2.0,
        "पण": 3.0, "कारण": 2.5, "विद्यार्थी": 3.0, "विद्यापीठ": 3.5,
        "अभ्यासक्रम": 3.5, "पर्यंत": 3.0, "नंतर": 3.0, "पूर्वी": 3.0,
        "मुलाखत": 3.0, "द्या": 2.5, "घ्या": 2.5, "करा": 3.0, "करून": 2.5,
        "मध्ये": 3.0, "हवे": 3.0, "हवी": 3.0, "नको": 3.0, "आहेत": 3.5,
        "केली": 2.5, "केले": 2.5, "भरा": 2.5, "पहा": 2.5, "पाहा": 2.5,
        "च्या": 3.0, "जागा": 2.0, "सुरुवात": 3.0, "असेल": 3.0, "असते": 3.0,
    },
    "raj-IN": {
        "म्हो": 3.5, "म्हूँ": 3.5, "म्हाणै": 3.5, "म्हाड़े": 3.5, "म्हारो": 3.5,
        "थारो": 3.5, "थारू": 3.5, "थानै": 3.5, "कोनी": 3.5, "आवै": 3.5,
        "करै": 3.0, "है": 1.0, "अठै": 3.5, "बतावो": 3.0, "रयो": 3.5,
        "काई": 3.0, "म्हारे": 3.5, "आप रो": 3.0, "सा": 1.5, "बातावो": 3.5,
    },
}

BENGALI_SCRIPT_MARKERS: dict[str, dict[str, float]] = {
    "bn-IN": {"আছে": 3.0, "আমি": 2.5, "কি": 2.0, "কত": 2.5, "এবং": 2.5, "না": 1.5,
              "ভর্তি": 2.0, "টাকা": 2.0, "করতে": 2.5, "চাই": 2.5},
    "as-IN": {"আছে": 3.0, "মই": 3.5, "কিমান": 3.5, "আৰু": 3.5, "নহয়": 3.0,
              "ভৰ্তি": 3.0, "টকা": 3.0, "কৰিব": 3.5},
}


@dataclass
class LIDResult:
    language: str
    confidence: float
    method: str = "lexicon"      # explicit_name | script | lexicon | acoustic | hybrid
    scores: dict[str, float] = field(default_factory=dict)
    candidates: tuple[str, ...] = ()
    detail: str = ""

    @property
    def is_confident(self) -> bool:
        from ...config import settings

        return self.confidence >= settings.lid_confidence_threshold

    @property
    def display(self) -> str:
        return get_language(self.language).english_name

    def to_dict(self) -> dict[str, object]:
        return {
            "language": self.language,
            "confidence": round(self.confidence, 3),
            "method": self.method,
            "scores": {k: round(v, 3) for k, v in sorted(
                self.scores.items(), key=lambda kv: -kv[1]
            )[:5]},
            "detail": self.detail,
        }


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def tokenize(text: str) -> list[str]:
    return [t for t in _WORD_RE.findall(text or "") if t]


def _script_profile(text: str) -> dict[str, float]:
    """Fraction of alphabetic characters per script block."""
    counts: dict[str, int] = {}
    total = 0
    for ch in text or "":
        if not ch.isalpha():
            continue
        total += 1
        cp = ord(ch)
        for name, low, high, _codes in SCRIPT_BLOCKS:
            if low <= cp <= high:
                counts[name] = counts.get(name, 0) + 1
                break
        else:
            if ("A" <= ch <= "Z") or ("a" <= ch <= "z"):
                counts["Latin"] = counts.get("Latin", 0) + 1
    if total == 0:
        return {}
    return {k: v / total for k, v in counts.items()}


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFC", (text or "").lower())
    return re.sub(r"[^\w\sऀ-ॿঀ-৿਀-੿઀-૿ୀ-୿஀-௿ఀ-౿ಀ-೿ഀ-ൿ؀-ۿ]", " ", text)


def _marker_score(tokens: list[str], raw: str, markers: dict[str, float]) -> float:
    score = 0.0
    lowered = [t.lower() for t in tokens]
    for marker, weight in markers.items():
        if " " in marker:
            score += weight * raw.lower().count(marker.lower())
            continue
        if marker.lower() in lowered:
            score += weight * lowered.count(marker.lower())
        elif len(marker) > 2 and marker.lower() in raw.lower():
            score += weight * 0.5
    return score


def _tiebreak_rank(code: str) -> int:
    """Order equal scores by the campus's regional language, not dict order.

    Hindi and Marathi share the Devanagari script, so an utterance whose words
    are in neither marker list scores identically for both. Before this, the
    winner was whichever code happened to come first in the candidate list,
    which silently sent Marathi callers to Hindi.
    """
    from ...config import settings

    return 0 if code == settings.devanagari_preference else 1


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #


def detect_language_name(text: str, candidates: tuple[str, ...] | None = None) -> LIDResult | None:
    """High-confidence path: the caller named the language."""
    raw = _norm(text)
    if not raw.strip():
        return None
    lowered = raw.strip().strip(".!,? ")
    hits: list[tuple[str, str, int]] = []
    for name, code in NAME_TO_CODE.items():
        if candidates and code not in candidates:
            continue
        if name in lowered.split() or (len(name) > 3 and name in lowered):
            hits.append((name, code, len(name)))
    if not hits:
        return None
    # longest name wins ("rajasthani" over "raj")
    hits.sort(key=lambda h: -h[2])
    best_name, best_code, _ = hits[0]
    scores: dict[str, float] = {}
    for _, code, length in hits:
        scores[code] = max(scores.get(code, 0.0), length / 10.0)
    return LIDResult(
        language=best_code,
        confidence=0.97,
        method="explicit_name",
        scores=scores,
        candidates=candidates or (),
        detail=f"caller named '{best_name}'",
    )


def detect_language_text(
    text: str,
    candidates: tuple[str, ...] | None = None,
    prefer_name_match: bool = True,
) -> LIDResult:
    """Full lexical detector. Never raises; worst case returns en-IN, conf 0."""
    allowed = tuple(candidates) if candidates else tuple(LANGUAGES.keys())
    if not (text or "").strip():
        return LIDResult(
            language="en-IN", confidence=0.0, method="empty", candidates=allowed,
            detail="no speech recognised",
        )

    if prefer_name_match:
        named = detect_language_name(text, allowed)
        if named:
            return named

    raw = _norm(text)
    tokens = tokenize(raw)
    profile = _script_profile(text)
    scores: dict[str, float] = {code: 0.0 for code in allowed}

    # --- script vote ------------------------------------------------------- #
    script_codes: list[str] = []
    script_share = 0.0
    for script_name, low, high, codes in SCRIPT_BLOCKS:
        share = profile.get(script_name, 0.0)
        if share >= 0.25:
            script_codes.extend(c for c in codes if c in allowed)
            script_share = max(script_share, share)
    latin_share = profile.get("Latin", 0.0)

    for code in script_codes:
        scores[code] = scores.get(code, 0.0) + 6.0 * script_share

    # Assamese vs Bengali: Assamese uses ৰ (U+09F0) / ৱ (U+09F1) which Bengali lacks
    if profile.get("Bengali", 0) > 0.25:
        if "\u09f0" in text or "\u09f1" in text:
            scores["as-IN"] = scores.get("as-IN", 0.0) + 8.0
            scores["bn-IN"] = max(0.0, scores.get("bn-IN", 0.0) - 4.0)
        for code, markers in BENGALI_SCRIPT_MARKERS.items():
            if code in allowed:
                scores[code] = scores.get(code, 0.0) + _marker_score(tokens, raw, markers)

    # --- Devanagari disambiguation ----------------------------------------- #
    if profile.get("Devanagari", 0) > 0.25:
        for code, markers in DEVANAGARI_MARKERS.items():
            if code in allowed:
                scores[code] = scores.get(code, 0.0) + _marker_score(tokens, raw, markers)

    # --- other scripts: use each language's marker list -------------------- #
    for code in allowed:
        if code in script_codes and code not in DEVANAGARI_MARKERS:
            markers = {m: 2.5 for m in get_language(code).markers}
            scores[code] = scores.get(code, 0.0) + _marker_score(tokens, raw, markers)

    # --- Latin script: English vs Hinglish vs Marathi vs Rajasthani --------- #
    if latin_share >= 0.25 or not script_codes:
        n = max(len(tokens), 1)
        hinglish = _marker_score(tokens, raw, {m: 2.0 for m in HINGLISH_MARKERS})
        marathi = _marker_score(
            tokens, raw, {m: 3.0 for m in HINGLISH_MARATHI_MARKERS}
        )
        rajasthani = _marker_score(
            tokens, raw, {m: 3.0 for m in HINGLISH_RAJASTHANI_MARKERS}
        )
        english_hits = _marker_score(
            tokens, raw, {m: 2.0 for m in LANGUAGES["en-IN"].markers}
        )
        scores["en-IN"] = scores.get("en-IN", 0.0) + english_hits * (1.0 + latin_share)
        scores["hi-IN"] = scores.get("hi-IN", 0.0) + (hinglish / n) * 12.0
        # Marathi outranks Hinglish on weight because the marker list is
        # Marathi-specific ("mala", "aahe", "pahije") while the Hinglish list
        # contains short words that also occur in romanised Marathi.
        if "mr-IN" in allowed:
            scores["mr-IN"] = scores.get("mr-IN", 0.0) + (marathi / n) * 16.0
        if "raj-IN" in allowed:
            scores["raj-IN"] = scores.get("raj-IN", 0.0) + (rajasthani / n) * 14.0
        # long English function words with no Indic markers → clearly English
        if hinglish == 0 and english_hits > 0:
            scores["en-IN"] += 6.0
        # A wholly Latin-script utterance with no romanised-Indic marker in it is
        # English. Without this floor a question whose words all sit outside the
        # marker list ("When do classes start?") scored zero everywhere and came
        # back with zero confidence, which re-prompts a caller who has already
        # said, in English, what they want. English is the only Latin-script
        # candidate here, so the script itself is the evidence.
        if latin_share >= 0.6 and not script_codes and hinglish == 0 and marathi == 0 \
                and rajasthani == 0:
            scores["en-IN"] = scores.get("en-IN", 0.0) + 6.0 * latin_share

    # --- normalise to a confidence ---------------------------------------- #
    for code in list(scores.keys()):
        if code not in allowed:
            scores.pop(code, None)
    if not scores:
        scores = {"en-IN": 1.0}
    total = sum(max(0.0, v) for v in scores.values())
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], _tiebreak_rank(kv[0])))
    top_code, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0

    if total <= 0:
        return LIDResult(
            language="en-IN", confidence=0.0, method="unknown", scores=scores,
            candidates=allowed, detail="no signal in transcript",
        )

    share = top_score / total
    # margin between the winner and the runner-up drives confidence
    margin = (top_score - second_score) / total if total else 0.0
    evidence = min(1.0, (top_score / 12.0))
    confidence = round(min(0.98, 0.45 * share + 0.4 * margin + 0.35 * evidence), 3)

    method = "script" if script_share >= 0.5 else "lexicon"
    detail = (
        f"script_share={script_share:.2f} latin={latin_share:.2f} "
        f"top={top_code}:{top_score:.1f} runner_up={ranked[1][0] if len(ranked)>1 else '-'}"
        f":{second_score:.1f} tokens={len(tokens)}"
    )
    return LIDResult(
        language=top_code,
        confidence=confidence,
        method=method,
        scores=scores,
        candidates=allowed,
        detail=detail,
    )


def romanised_language_hint(text: str) -> str | None:
    """Quick check used while the caller is still speaking Latin-script Hinglish."""
    result = detect_language_text(text)
    return result.language if result.confidence > 0.5 else None
