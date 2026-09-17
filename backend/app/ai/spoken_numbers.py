"""Spoken number formatting for Indian telephone speech.

TTS engines read "₹150000" as "one hundred fifty thousand" (US grouping) or
worse. Callers think in lakh/crore. This module renders money and counts the way
an admissions counsellor in Dhule says them — in English, Hindi, Marathi and
Rajasthani.

Marathi and Hindi share the Devanagari script but not the number words, so each
has its own unit table: 100 000 is "लाख" in both, but the fallback phrasing and
the units below it differ ("शंभर" vs "सौ", "वर्षे" vs "साल").
"""

from __future__ import annotations

import re

_ONES_EN = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS_EN = ["zero", "ten", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

_ONES_HI = ["शून्य", "एक", "दो", "तीन", "चार", "पाँच", "छह", "सात", "आठ", "नौ", "दस", "ग्यारह", "बारह", "तेरह", "चौदह", "पंद्रह", "सोलह", "सत्रह", "अठारह", "उन्नीस"]

_HI_UNITS = [
    (10_000_000, "करोड़"),
    (100_000, "लाख"),
    (1_000, "हज़ार"),
    (100, "सौ"),
]

_MR_UNITS = [
    (10_000_000, "कोटी"),
    (100_000, "लाख"),
    (1_000, "हजार"),
    (100, "शंभर"),
]
_EN_UNITS = [
    (10_000_000, "crore"),
    (100_000, "lakh"),
    (1_000, "thousand"),
    (100, "hundred"),
]


def number_to_words_en(number: float) -> str:
    """Indian grouping: 150000 -> 'one lakh fifty thousand'."""
    if number is None:
        return ""
    value = int(round(float(number)))
    negative = value < 0
    value = abs(value)
    if value == 0:
        return "zero"
    parts: list[str] = []
    for unit_value, unit_name in _EN_UNITS:
        if value >= unit_value:
            count = value // unit_value
            value -= count * unit_value
            parts.append(f"{_two_digit_en(count)} {unit_name}")
    if value:
        parts.append(_two_digit_en(value))
    text = " ".join(parts)
    return ("minus " + text) if negative else text


def _two_digit_en(value: int) -> str:
    if value < 20:
        return _ONES_EN[value]
    tens, ones = divmod(value, 10)
    return f"{_TENS_EN[tens]}" + (f" {_ONES_EN[ones]}" if ones else "")


def number_to_words_hi(number: float) -> str:
    """Hindi: digits for the components (TTS reads them correctly) + lakh/crore words."""
    if number is None:
        return ""
    value = int(round(float(number)))
    if value == 0:
        return "शून्य"
    parts: list[str] = []
    for unit_value, unit_name in _HI_UNITS:
        if value >= unit_value:
            count = value // unit_value
            value -= count * unit_value
            parts.append(f"{count} {unit_name}")
    if value:
        parts.append(str(value))
    return " ".join(parts)


def number_to_words_mr(number: float) -> str:
    """Marathi: digits for the components (TTS reads them correctly) + Marathi units.

    150000 -> '1 लाख 50 हजार'; 4 -> '4'.
    """
    if number is None:
        return ""
    value = int(round(float(number)))
    if value == 0:
        return "शून्य"
    parts: list[str] = []
    for unit_value, unit_name in _MR_UNITS:
        if value >= unit_value:
            count = value // unit_value
            value -= count * unit_value
            parts.append(f"{count} {unit_name}")
    if value:
        parts.append(str(value))
    return " ".join(parts)


def parse_money(value: object) -> float | None:
    """Accept 150000, '150000', '1.5 lakh', '₹6,00,000', 'INR 1.2 crore'."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lower().replace(",", "").replace("₹", "").replace("inr", "")
    multiplier = 1.0
    if "crore" in text or " cr" in text or text.endswith("cr"):
        multiplier = 10_000_000
        text = re.sub(r"crore|cr", "", text)
    elif "lakh" in text or "lac" in text:
        multiplier = 100_000
        text = re.sub(r"lakh|lac", "", text)
    elif "thousand" in text:
        multiplier = 1_000
        text = text.replace("thousand", "")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0)) * multiplier
    except ValueError:
        return None


def spoken_money(value: object, language: str = "en-IN", *, per: str | None = None) -> str:
    """Render a fee as it should be spoken. Returns '' when there is nothing."""
    amount = parse_money(value)
    if amount is None:
        return ""
    base = language.split("-")[0]
    if base == "mr":
        text = f"{number_to_words_mr(amount)} रुपये"
    elif base == "hi" or language == "raj-IN":
        text = f"{number_to_words_hi(amount)} रुपये"
    else:
        text = f"{number_to_words_en(amount)} rupees"
    if per:
        suffix = {"year": "per year", "total": "in total", "month": "per month"}.get(per, per)
        if base == "mr":
            suffix = {
                "year": "प्रति वर्ष",
                "total": "संपूर्ण अभ्यासक्रमासाठी",
                "month": "प्रति महिना",
            }.get(per, per)
        elif base == "hi" or language == "raj-IN":
            suffix = {"year": "प्रति वर्ष", "total": "पूरे कोर्स के", "month": "प्रति माह"}.get(
                per, per
            )
        text = f"{text} {suffix}"
    return text


def spoken_count(value: object, noun: str = "", language: str = "en-IN") -> str:
    amount = parse_money(value)
    if amount is None:
        return ""
    base = language.split("-")[0]
    if base in {"hi", "mr"} or language == "raj-IN":
        # Devanagari-script languages: TTS reads the digits natively, and a
        # spelled-out count ("one hundred twenty seats") is never said on a call.
        return f"{int(amount)} {noun}".strip()
    return f"{number_to_words_en(amount)} {noun}".strip()


def spoken_year_range(value: object) -> str:
    text = str(value or "").strip()
    return text.replace("-", " to ") if re.fullmatch(r"\d{4}-\d{2}", text) else text


def duration_phrase(years: object, language: str = "en-IN") -> str:
    amount = parse_money(years)
    if amount is None:
        return ""
    base = language.split("-")[0]
    count = int(amount)
    if base == "mr":
        # "1 वर्ष" but "4 वर्षे" — Marathi pluralises the noun, not the number.
        return f"{count} वर्ष" if count == 1 else f"{count} वर्षे"
    if base == "hi" or language == "raj-IN":
        return f"{count} साल" if count != 1 else "1 साल"
    if count == 1:
        return "one year"
    if count == 2:
        return "two years"
    return f"{number_to_words_en(count)} years"
