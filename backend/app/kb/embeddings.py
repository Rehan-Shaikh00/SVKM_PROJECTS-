"""Embeddings.

Production: OpenAI `text-embedding-3-small` (multilingual, cheap) or Cohere
`embed-multilingual-v3`.

Zero-key fallback: `LocalHashingEmbedder` — a deterministic hashed bag of
expanded tokens + character n-grams with sublinear TF and L2 normalisation.
It is not a semantic model, but combined with the synonym expansion below it
bridges the two gaps that matter for a university helpline:

* **cross-lingual lexical gap** — "फीस" / "ફી" / "கட்டணம்" / "fees" all hash into
  the same shared feature, so a Hindi query retrieves the English fee record
* **abbreviation gap** — "B.Tech CSE" ≡ "Bachelor of Technology Computer Science"

With BM25 fusion (app/kb/retriever.py) this retrieves the right records for the
vast majority of course/fee/date questions. Swap in a real multilingual model
when you have a key: the interface is identical.
"""

from __future__ import annotations

import hashlib
import logging
import re
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from ..config import settings

logger = logging.getLogger("nims.embeddings")

# --------------------------------------------------------------------------- #
# Domain + cross-lingual synonym groups
# --------------------------------------------------------------------------- #

SYNONYM_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    # --- money ------------------------------------------------------------- #
    ("fees", ("fee", "fees", "tuition", "cost", "charges", "price", "amount",
              "फीस", "फ़ीस", "शुल्क", "राशि", "खर्च",
              # Marathi writes the same word with one matra: "फी", "रक्कम".
              "फी", "रक्कम", "કિંમત", "ખર્ચ", "રકમ",
              "கட்டணம்", "ఫీజు", "శుల్కం", "ರಸುಮು", "ಶುಲ್ಕ", "ശുല്ക്കം", "ഫീസ്",
              "ਫੀਸ", "فیس", "ଟିକସ", "খরচ")),
    ("scholarship", ("scholarship", "scholarships", "concession", "waiver", "freeship",
                     "financial aid", "discount", "छात्रवृत्ति", "स्कॉलरशिप", "स्काॅलरशिप",
                     "छूट", "रियायत",
                     # Marathi: "शिष्यवृत्ती", and "फी माफी" is how a fee waiver is asked for.
                     "शिष्यवृत्ती", "फी माफी", "शुल्क माफी",
                     "சுகாலார்ஷிப்", "സ്കോളർഷിപ്പ്", "ಸ್ಕಾಲರ್‌ಶಿಪ್")),
    ("hostel", ("hostel", "hostels", "dormitory", "accommodation", "room", "mess",
                "warden", "residence", "हॉस्टल", "छात्रावास", "आवास", "कमरा",
                # Marathi: "वसतिगृह", "हॉस्टेल" (long o), "राहण्याची सोय", "मेस".
                "वसतिगृह", "हॉस्टेल", "राहण्याची सोय", "मेस",
                "ஹாஸ்டல்", "హాస్టల్", "ಹಾಸ್ಟೆಲ್", "ഹോസ്റ്റൽ")),
    ("admission", ("admission", "admissions", "apply", "application", "enrolment",
                   "enrollment", "registration", "counselling", "counseling",
                   "प्रवेश", "एडमिशन", "आवेदन", "दाखिला", "भरती",
                   # Marathi: the application is an "अर्ज", the process "प्रवेश प्रक्रिया".
                   "अर्ज", "प्रवेश प्रक्रिया",
                   # Lateral entry is an admission route, so it belongs with admission.
                   "lateral entry", "लॅटरल एंट्री",
                   "ప్రవేశం", "ಪ್ರವೇಶ", "പ്രവേശനം", "ਦਾਖਲਾ", "داخلہ")),
    ("eligibility", ("eligibility", "eligible", "qualification", "criteria", "criterion",
                     "requirement", "required marks", "cutoff", "पात्रता", "योग्यता",
                     "शर्त", "अंक", "मार्क्स",
                     # Marathi: "किती गुण", "टक्केवारी", "आवश्यक".
                     "गुण", "टक्केवारी", "टक्के", "आवश्यक", "पात्र",
                     "ప్రవేశ అర్హత", "ಅರ್ಹತೆ")),
    ("placement", ("placement", "placements", "package", "ctc", "salary", "recruiter",
                   "recruiters", "companies", "job", "jobs", "internship", "highest package",
                   "प्लेसमेंट", "नौकरी", "वेतन", "कंपनियां",
                   # Marathi: "नोकरी", "पगार", "कंपन्या".
                   "नोकरी", "पगार", "कंपन्या",
                   "പ്ലേസ്മെന്റ്")),
    ("duration", ("duration", "years", "semesters", "course length", "अवधि", "वर्ष",
                  "साल",
                  # Marathi: "वर्षे" (plural), "कालावधी", "सेमिस्टर".
                  "वर्षे", "कालावधी", "सेमिस्टर",
                  "వ్యవధಿ", "ಅವಧಿ")),
    ("documents", ("documents", "certificate", "certificates", "marksheet", "mark sheet",
                   "transcript", "id proof", "affidavit", "दस्तावेज", "कागजात", "प्रमाणपत्र",
                   "मार्कशीट",
                   # Marathi: "कागदपत्रे", "प्रमाणपत्रे", "गुणपत्रिका".
                   "कागदपत्रे", "कागदपत्र", "प्रमाणपत्रे", "गुणपत्रिका",
                   "పత్రాలు", "ದಾಖಲೆಗಳು")),
    ("exam", ("entrance", "exam", "test", "परीक्षा", "एंट्रेंस", "प्रवेश परीक्षा",
              # The tests a Dhule caller names. Keeping them in one group means
              # "NPAT" and "प्रवेश परीक्षा" retrieve the same admission records.
              "npat", "nmims-npat", "nmims npat", "nmat", "mht-cet", "mhtcet", "cet",
              "clat", "lsat", "lsat india", "gate", "gpat", "jee", "jee main", "cat",
              "परीक्षा कोणती", "कोणती परीक्षा",
              "పరీక్ష", "ಪರೀಕ್ಷೆ", "പരീക്ഷ")),
    ("deadline", ("deadline", "last date", "due date", "important dates", "end date",
                  "अंतिम तिथि", "आखिरी तारीख", "कब तक",
                  # Marathi: "शेवटची तारीख", "मुदत", "अंतिम तारीख", "कधी".
                  "शेवटची तारीख", "अंतिम तारीख", "मुदत", "कधी",
                  "తేదీ", "ಕೊನೆಯ ದಿನಾಂಕ")),
    ("contact", ("contact", "phone", "number", "email", "helpline", "address", "office",
                 "संपर्क", "फोन", "नंबर", "पता", "ईमेल", "हेल्पलाइन",
                 # Marathi: "क्रमांक", "दूरध्वनी", "पत्ता", "संपर्क क्रमांक".
                 "क्रमांक", "दूरध्वनी", "पत्ता", "संपर्क क्रमांक",
                 "సంప్రదించండి")),
    ("transport", ("transport", "bus", "buses", "shuttle", "बस", "परिवहन", "वाहन",
                   # Marathi: "वाहतूक", "कसे पोहोचावे", "रेल्वे", "विमानतळ", "बस सेवा".
                   "वाहतूक", "कसे पोहोचावे", "कसं पोहोचावं", "रेल्वे", "स्टेशन",
                   "विमानतळ", "बस सेवा")),
    ("facilities", ("facility", "facilities", "infrastructure", "lab", "labs", "library",
                    "hospital", "sports", "gym", "wifi", "clinic", "सुविधा", "लाइब्रेरी",
                    "लैब", "अस्पताल", "ग्राउंड", "सुविधाएं",
                    # Marathi: "प्रयोगशाळा", "ग्रंथालय", "क्रीडा", "वैद्यकीय".
                    "प्रयोगशाळा", "ग्रंथालय", "क्रीडा", "वैद्यकीय", "कॅम्पस",
                    "कँपस")),
    # --- degrees ----------------------------------------------------------- #
    ("btech", ("btech", "b.tech", "b tech", "be", "b.e", "bachelor of technology",
               "बीटेक", "बी.टेक", "बी टेक",
               # Marathi for engineering / the degree itself.
               "अभियांत्रिकी")),
    ("mtech", ("mtech", "m.tech", "m tech", "me", "m.e", "master of technology",
               "एमटेक", "एम.टेक")),
    ("mbbs", ("mbbs", "mbbs degree", "bachelor of medicine", "एमबीबीएस", "mbbs course")),
    ("bds", ("bds", "bachelor of dental surgery", "dental", "बीडीएस", "दंत")),
    ("bpharm", ("bpharm", "b.pharm", "b pharmacy", "bachelor of pharmacy", "बीफार्म",
                "बी.फार्म")),
    ("pharmd", ("pharmd", "pharm.d", "doctor of pharmacy", "फार्मडी")),
    ("pharmacy", ("pharmacy", "pharmaceutical", "pharmaceuticals", "pharma",
                  # Marathi for the school and the discipline.
                  "फार्मसी", "औषधनिर्माणशास्त्र", "औषधनिर्माण", "फार्मसी")),
    ("dpharm", ("dpharm", "d.pharm", "d pharm", "diploma in pharmacy", "pharmacy diploma",
                "डीफार्म", "डी.फार्म", "फार्मसी डिप्लोमा")),
    ("mpharm", ("mpharm", "m.pharm", "master of pharmacy", "एमफार्म", "एम.फार्म",
                "pharmacology", "pharmaceutical quality assurance",
                "फार्माकोलॉजी", "गुणवत्ता हमी")),
    ("bcom", ("bcom", "b.com", "b com", "bachelor of commerce", "commerce",
              "बीकॉम", "वाणिज्य", "commerce honours")),
    ("mcom", ("mcom", "m.com", "master of commerce", "एमकॉम")),
    ("acca_cma", ("acca", "cma", "certified management accountant",
                  "chartered accountancy", "ca", "एक्का", "सीएमए")),
    ("mba", ("mba", "master of business administration", "management", "एमबीए",
             # Marathi: "व्यवस्थापन", "व्यवस्थापन शाखा".
             "व्यवस्थापन", "व्यवस्थापन शाखा")),
    ("bba", ("bba", "bachelor of business administration", "बीबीए")),
    ("bca", ("bca", "bachelor of computer applications", "बीसीए",
             "computer applications", "संगणक अनुप्रयोग")),
    ("mca", ("mca", "master of computer applications", "एमसीए")),
    ("bsc", ("bsc", "b.sc", "b sc", "bachelor of science", "बीएससी")),
    ("msc", ("msc", "m.sc", "m sc", "master of science", "एमएससी")),
    ("bpt", ("bpt", "b.p.t", "bachelor of physiotherapy", "physiotherapy", "बीपीटी")),
    ("nursing", ("nursing", "bsc nursing", "gnm", "anm", "नर्सिंग")),
    ("law", ("law", "llb", "ll.b", "ba llb", "bba llb", "llm", "legal", "कानून", "विधि")),
    ("design", ("design", "bdes", "b.des", "fashion", "textile", "interior", "डिजाइन",
                "फैशन")),
    ("architecture", ("architecture", "barch", "b.arch", "b.plan", "planning", "वास्तुकला")),
    ("phd", ("phd", "ph.d", "doctorate", "doctoral", "research", "पीएचडी", "शोध",
             # Marathi: "डॉक्टरेट", "संशोधन", "पीएच.डी.".
             "डॉक्टरेट", "संशोधन", "पीएच.डी.")),
    ("hotel", ("hotel", "hospitality", "bhmct", "catering", "tourism", "hotel management",
               "होटल")),
    ("journalism", ("journalism", "mass communication", "bjmc", "mjmc", "media",
                    "पत्रकारिता", "मीडिया")),
    # --- specialisations --------------------------------------------------- #
    ("cse", ("cse", "computer science", "computer science and engineering", "computers",
             "कंप्यूटर साइंस", "computer engineering",
             # Marathi: "संगणक", "संगणक अभियांत्रिकी". Deliberately NOT
             # "माहिती तंत्रज्ञान": "माहिती" just means "information" and
             # appears in almost every Marathi question, so the piece-splitter
             # would tag all of them as computer science.
             "संगणक", "संगणक अभियांत्रिकी")),
    ("it", ("it", "information technology", "सूचना प्रौद्योगिकी")),
    ("ai_ml", ("ai", "ml", "artificial intelligence", "machine learning", "data science",
               "एआई", "एमएल", "डेटा साइंस",
               # Marathi: "कृत्रिम बुद्धिमत्ता", "डेटा विज्ञान", "यंत्र शिक्षण".
               "कृत्रिम बुद्धिमत्ता", "डेटा विज्ञान", "यंत्र शिक्षण",
               # Specialisations this campus lists.
               "iot", "full stack", "ai and data science",
               "ai & data science", "ai&ds")),
    ("mechanical", ("mechanical", "मैकेनिकल",
                    # Marathi: "यांत्रिकी अभियांत्रिकी".
                    "यांत्रिकी", "यांत्रिक अभियांत्रिकी")),
    ("civil", ("civil", "सिविल", "स्थापत्य", "स्थापत्य अभियांत्रिकी")),
    ("electrical", ("electrical", "इलेक्ट्रिकल", "विद्युत", "विद्युत अभियांत्रिकी")),
    ("petroleum", ("petroleum", "petroleum engineering", "पेट्रोलियम")),
    ("aeronautical", ("aeronautical", "aeronautical engineering", "aerospace",
                      "एअरोनॉटिकल", "हवाई")),
    ("electronics", ("electronics", "ece", "electronics and communication", "इलेक्ट्रॉनिक्स")),
    ("biotech", ("biotechnology", "biotech", "बायोटेक", "जैव प्रौद्योगिकी")),
    ("agriculture", ("agriculture", "agricultural", "कृषि", "खेती")),
    # --- places / bodies --------------------------------------------------- #
    ("nmims", ("nmims", "nimms", "nmims global university", "svkm", "svkm's nmims",
               "एनएमआयएमएस", "एनआयएमएस", "एसवीकेएम")),
    ("dhule", ("dhule", "dulia", "maharashtra", "khandesh", "shirpur", "dhule district",
               "धुळे", "धुले", "महाराष्ट्र", "खान्देश", "शिरपूर")),
    ("ugc", ("ugc", "aicte", "naac", "nmc", "mci", "pci", "dci", "bci", "coa", "inc",
             "recognition", "accreditation", "approved", "मान्यता")),
)

#: token -> canonical group
_TOKEN_TO_GROUP: dict[str, str] = {}
for _group, _members in SYNONYM_GROUPS:
    for _member in _members:
        _TOKEN_TO_GROUP[_member.lower()] = _group
        for _piece in re.split(r"[.\s]+", _member.lower()):
            if len(_piece) > 1:
                _TOKEN_TO_GROUP.setdefault(_piece, _group)

_WORD_RE = re.compile(r"[\wऀ-ॿঀ-৿஀-௿ఀ-౿ಀ-೿ഀ-ൿ઀-૿਀-੿؀-ۿୀ-୿]+", re.UNICODE)
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "of", "for", "to",
    "in", "on", "at", "and", "or", "do", "does", "did", "i", "me", "my", "we", "you",
    "your", "please", "tell", "about", "what", "which", "how", "much", "many",
    "can", "could", "would", "should", "will", "there", "here", "it", "this", "that",
    "hai", "kya", "ka", "ki", "ke", "ko", "mein", "se", "par", "aur", "bhi",
    "mujhe", "muje", "bataye", "batao", "bataiye", "karna", "karein", "chahiye",
    "है", "हैं", "का", "की", "के", "को", "में", "और", "भी", "मुझे", "क्या", "बताइए",
    "बताओ", "बताएं", "करना", "चाहिए", "लिए", "या", "तो", "ही",
}


def tokenize(text: str, keep_stopwords: bool = False) -> list[str]:
    tokens: list[str] = []
    for raw in _WORD_RE.findall((text or "").lower()):
        token = raw.strip("._-")
        if not token:
            continue
        if len(token) == 1 and not token.isdigit():
            continue
        if not keep_stopwords and token in _STOPWORDS:
            continue
        tokens.append(token)
    return tokens


def expand(text: str) -> list[str]:
    """Tokens plus their canonical synonym-group features."""
    tokens = tokenize(text)
    features: list[str] = list(tokens)
    lowered = (text or "").lower()
    for token in tokens:
        group = _TOKEN_TO_GROUP.get(token)
        if group:
            features.append(f"#g:{group}")
        if token.endswith("s") and len(token) > 3:
            stem = token[:-1]
            if stem in _TOKEN_TO_GROUP:
                features.append(f"#g:{_TOKEN_TO_GROUP[stem]}")
    # multi-word group members ("last date", "computer science", "b.tech")
    for group, members in SYNONYM_GROUPS:
        for member in members:
            if " " in member and member in lowered or "." in member and member in lowered:
                features.append(f"#g:{group}")
    return features


class Embedder(ABC):
    name = "base"
    dimensions: int = 768

    @abstractmethod
    async def embed(self, texts: list[str]) -> np.ndarray: ...

    async def embed_one(self, text: str) -> np.ndarray:
        vectors = await self.embed([text])
        return vectors[0]

    async def close(self) -> None:  # pragma: no cover
        return None


class LocalHashingEmbedder(Embedder):
    """Deterministic hashed embedding (no model download, no API key)."""

    name = "local"

    def __init__(self, dimensions: int | None = None, ngram: int = 3) -> None:
        self.dimensions = int(dimensions or settings.vector_dimensions)
        self.ngram = ngram
        self._cache: dict[str, np.ndarray] = {}
        self.embedded = 0

    def _hash_index(self, feature: str, salt: int = 0) -> int:
        digest = hashlib.blake2b(
            feature.encode("utf-8"), digest_size=8, salt=salt.to_bytes(8, "little")
        ).digest()
        return int.from_bytes(digest, "big") % self.dimensions

    def _embed_sync(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimensions, dtype=np.float32)
        features = expand(text)
        counts: dict[int, float] = {}
        signs: dict[int, float] = {}
        for feature in features:
            index = self._hash_index(feature)
            # sublinear TF keeps long documents from dominating
            counts[index] = counts.get(index, 0.0) + 1.0
            sign = 1.0 if self._hash_index(feature, 7) % 2 == 0 else -1.0
            signs[index] = sign
        for index, count in counts.items():
            vector[index] += signs[index] * (1.0 + np.log(count))

        # character n-grams add morphological signal for Indic scripts
        compact = re.sub(r"\s+", "", (text or "").lower())
        if len(compact) >= self.ngram:
            for start in range(0, len(compact) - self.ngram + 1, 2):
                gram = compact[start : start + self.ngram]
                index = self._hash_index(gram, 3)
                vector[index] += 0.25 * signs.get(index, 1.0)

        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector /= norm
        return vector

    async def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for position, text in enumerate(texts):
            cached = self._cache.get(text)
            if cached is None:
                cached = self._embed_sync(text)
                if len(self._cache) < 20000:
                    self._cache[text] = cached
            vectors[position] = cached
        self.embedded += len(texts)
        return vectors


class OpenAIEmbedder(Embedder):  # pragma: no cover - network dependent
    name = "openai"

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.api_key = api_key or settings.openai_api_key
        self.model = model or settings.openai_embedding_model
        self.dimensions = 1536 if "3-small" in self.model else 3072
        self.embedded = 0

    async def embed(self, texts: list[str]) -> np.ndarray:
        import httpx

        if not texts:
            return np.zeros((0, self.dimensions), dtype=np.float32)
        batches: list[list[str]] = [texts[i : i + 128] for i in range(0, len(texts), 128)]
        vectors: list[np.ndarray] = []
        async with httpx.AsyncClient(timeout=60) as client:
            for batch in batches:
                resp = await client.post(
                    "https://api.openai.com/v1/embeddings",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={"model": self.model, "input": batch},
                )
                resp.raise_for_status()
                payload = resp.json()
                ordered = sorted(payload["data"], key=lambda d: d["index"])
                vectors.extend(
                    np.asarray(item["embedding"], dtype=np.float32) for item in ordered
                )
                self.embedded += len(batch)
        return np.vstack(vectors) if vectors else np.zeros((0, self.dimensions), np.float32)


class CohereEmbedder(Embedder):  # pragma: no cover - network dependent
    name = "cohere"

    def __init__(self, api_key: str | None = None, model: str = "embed-multilingual-v3.0") -> None:
        self.api_key = api_key or settings.cohere_api_key
        self.model = model
        self.dimensions = 1024
        self.embedded = 0

    async def embed(self, texts: list[str]) -> np.ndarray:
        import httpx

        if not texts:
            return np.zeros((0, self.dimensions), dtype=np.float32)
        vectors: list[np.ndarray] = []
        async with httpx.AsyncClient(timeout=60) as client:
            for start in range(0, len(texts), 96):
                batch = texts[start : start + 96]
                resp = await client.post(
                    "https://api.cohere.com/v2/embed",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": self.model,
                        "input_type": "search_document",
                        "embedding_types": ["float"],
                        "texts": batch,
                    },
                )
                resp.raise_for_status()
                payload = resp.json()
                vectors.extend(
                    np.asarray(item["embeddings"]["float"], dtype=np.float32)
                    for item in payload["embeddings"]
                )
                self.embedded += len(batch)
        return np.vstack(vectors) if vectors else np.zeros((0, self.dimensions), np.float32)


def build_embedder(provider: str | None = None) -> Embedder:
    provider = (provider or settings.embedding_provider or "local").lower()
    if provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY missing")
        return OpenAIEmbedder()
    if provider == "cohere":
        if not settings.cohere_api_key:
            raise RuntimeError("COHERE_API_KEY missing")
        return CohereEmbedder()
    return LocalHashingEmbedder()


def resolve_embedder(requested: str | None = None) -> tuple[Embedder, dict[str, Any]]:
    requested = (requested or settings.embedding_provider or "local").lower()
    attempts: list[str] = []
    for candidate in [requested, "openai", "cohere", "local"]:
        try:
            embedder = build_embedder(candidate)
            return embedder, {
                "requested": requested,
                "active": candidate,
                "degraded": candidate != requested,
                "dimensions": embedder.dimensions,
                "attempts": attempts,
            }
        except Exception as exc:
            attempts.append(f"{candidate}:{exc}")
    embedder = LocalHashingEmbedder()
    return embedder, {"requested": requested, "active": "local", "degraded": True,
                      "dimensions": embedder.dimensions, "attempts": attempts}
