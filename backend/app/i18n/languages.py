"""Language registry.

One place that knows, for every language the helpline serves:

* the BCP-47 code, English + native names
* which ASR / LID locales to request
* which TTS voices to use per provider
* the scripted prompts (greeting, language selection, no-input, escalation …)
* fallback behaviour (e.g. Rajasthani/Marwari has no mainstream ASR model, so we
  detect it separately but recognise + synthesise with Hindi voices)

Adding a language = add an entry here + enable it in SUPPORTED_LANGUAGES.
No code changes anywhere else.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Language:
    code: str                      # BCP-47, e.g. "hi-IN"
    english_name: str
    native_name: str
    #: locale handed to the ASR engine
    asr_locale: str = ""
    #: locale handed to the TTS engine (defaults to `code`)
    tts_locale: str = ""
    #: spoken language-ID services cannot distinguish these from `asr_locale`
    #: (e.g. Rajasthani is recognised as Hindi) — detection is lexical instead.
    lid_supported: bool = True
    #: language we fall back to for ASR/TTS when this one is unsupported
    fallback_code: str = ""
    voices: dict[str, str] = field(default_factory=dict)
    scripts: dict[str, str] = field(default_factory=dict)
    #: sample utterances used to tune the lexical detector + tests
    markers: tuple[str, ...] = ()


# --- scripted prompts ------------------------------------------------------
# Keep every prompt SHORT and speakable. `{{}}` placeholders are filled at run time.

EN = {
    "greeting": "Thank you for calling NMIMS Global University, Dhule. You are speaking with Saarthi, an AI assistant.",
    "language_prompt": "Please tell me your preferred language. कृपया अपनी भाषा बताइए। कृपया आपली भाषा सांगा.",
    "language_confirm": "Great, I will continue in {{language}}.",
    "language_unclear": "Sorry, I did not catch that. Please say the name of your language, for example English, Hindi, or Marathi.",
    "language_unsupported": "I am sorry, I cannot speak {{language}} yet. I can continue in English, Hindi or Marathi, or connect you to a human agent.",
    "menu": "You can ask me about courses, admission, fees, eligibility, scholarships, hostels, placements, or how to reach the campus. What would you like to know?",
    "no_input_1": "I am still here. Please ask your question.",
    "no_input_2": "I did not hear anything. Would you like me to connect you to the admissions team?",
    "thinking": "Let me check that.",
    "not_found": "I do not have that information right now. Let me connect you to a human.",
    "low_confidence": "I am not fully certain about that, so let me connect you to the admissions team.",
    "escalate_offer": "Would you like me to connect you to a person?",
    "escalating": "Connecting you to the admissions team now. Please stay on the line.",
    "queue_hold": "All our advisors are busy. You are in the queue, please stay on the line.",
    "farewell": "Thank you for calling NMIMS Global University, Dhule. Goodbye.",
    # Spoken whenever a follow-up is offered, long answer or short, so it must not
    # claim the answer was long.
    "followup_offer": "Should I send these details to you by SMS or email?",
    "barge_in_ack": "Go ahead.",
    "human_requested": "Of course, let me get a person for you.",
    "recording_consent": "For quality and training, this call may be recorded. If you prefer, you can ask us not to record.",
    "dtmf_fallback": "If it is easier, press one for English, two for Hindi, or stay on the line and speak again.",
}

HI = {
    "greeting": "एनएमआयएमएस ग्लोबल यूनिवर्सिटी, धुले में कॉल करने के लिए धन्यवाद। यह कॉल एक एआई सहायक द्वारा उत्तर दी जा रही है।",
    "language_prompt": "कृपया अपनी भाषा बताइए। Please tell me your preferred language.",
    "language_confirm": "ठीक है, मैं अब {{language}} में बात करूँगा।",
    "language_unclear": "माफ़ कीजिए, मैं समझ नहीं पाया। कृपया अपनी भाषा का नाम बोलिए, जैसे हिंदी या इंग्लिश।",
    "language_unsupported": "क्षमा करें, मुझे {{language}} नहीं आती। मैं हिंदी, अंग्रेज़ी या मराठी में मदद कर सकता हूँ, या किसी व्यक्ति से बात करा सकता हूँ।",
    "menu": "आप कोर्स, एडमिशन, फीस, पात्रता, छात्रवृत्ति, हॉस्टल, प्लेसमेंट या कैंपस के बारे में पूछ सकते हैं। बताइए क्या जानना है?",
    "no_input_1": "मैं यहीं हूँ। कृपया अपना प्रश्न पूछिए।",
    "no_input_2": "मुझे कुछ सुनाई नहीं दिया। क्या मैं आपको एडमिशन टीम से जोड़ दूँ?",
    "thinking": "एक पल, मैं देख रहा हूँ।",
    "not_found": "मुझे यह जानकारी अभी उपलब्ध नहीं है। मैं आपको किसी व्यक्ति से जोड़ता हूँ।",
    "low_confidence": "मैं इस बारे में पूरी तरह निश्चित नहीं हूँ, इसलिए मैं आपको एडमिशन टीम से जोड़ता हूँ।",
    "escalate_offer": "क्या आप किसी व्यक्ति से बात करना चाहेंगे?",
    "escalating": "अब आपको एडमिशन टीम से जोड़ा जा रहा है। कृपया लाइन पर बने रहिए।",
    "queue_hold": "हमारे सभी सलाहकार व्यस्त हैं। आप कतार में हैं, कृपया लाइन पर बने रहिए।",
    "farewell": "एनएमआयएमएस ग्लोबल यूनिवर्सिटी, धुले में कॉल करने के लिए धन्यवाद। नमस्ते।",
    "followup_offer": "क्या मैं यह विवरण आपको एसएमएस या ईमेल से भेज दूँ?",
    "barge_in_ack": "जी, बोलिए।",
    "human_requested": "ज़रूर, मैं आपके लिए किसी व्यक्ति को बुलाता हूँ।",
    "recording_consent": "गुणवत्ता जाँच के लिए यह कॉल रिकॉर्ड की जा सकती है।",
    "dtmf_fallback": "यदि आसान हो तो इंग्लिश के लिए एक और हिंदी के लिए दो दबाइए, या दोबारा बोलकर बताइए।",
}

RAJ = {
    "greeting": "एनएमआयएमएस ग्लोबल यूनिवर्सिटी, धुले में फोन करण खातर धन्यवाद। बात एआई सहायक कर री है।",
    "language_prompt": "आप अपणी भाषा बतावो। Please tell me your preferred language.",
    "language_confirm": "ठिक है, मैं {{language}} में बात करूँगा।",
    "language_unclear": "माफ करना, म्हाने समझ नाई आयो। अपणी भाषा को नाम बोलो, जिम अंग्रेजी, हिंदी या राजस्थानी।",
    "language_unsupported": "माफ करना, म्हाणे {{language}} कोनी आवै। मैं हिंदी या अंग्रेजी में मदद कर सकूँ, या किसी माणस सूं मिला दूँ।",
    "menu": "थारूँ कोर्स, एडमिशन, फीस, पात्रता, स्कॉलरशिप, हॉस्टल या प्लेसमेंट बारे पूछणो हो तो बोलो। काई जाणणो है?",
    "no_input_1": "म्हो अठै हूँ। थारूँ सवाल बोलो।",
    "no_input_2": "म्हाणे केळायो नाईं। म्हूँ थानै एडमिशन टीम सूं मिला दूँ?",
    "thinking": "एक मिन्ट, म्हो देख रयो हूँ।",
    "not_found": "म्हाड़ै ई जाणकारी कोनी है। म्हूँ थानै किसी माणस सूं मिला दूँ।",
    "low_confidence": "म्हूँ पक्को कोनी हूँ, इरलियाँ थानै एडमिशन टीम सूं मिला दूँ।",
    "escalate_offer": "थानै किसी माणस सूं बात करणी है?",
    "escalating": "थानै एडमिशन टीम सूं जोड़ रयो हूँ। लाईन पर बण्या रहजो।",
    "queue_hold": "सारा सलाहकार बिजी है। थोड़ो ठहरो, लाईन पर बण्या रहजो।",
    "farewell": "धन्यवाद। राम राम सा।",
    "followup_offer": "म्हूँ ई बात थानै एसएमएस या ईमेल सूं भेज दूँ?",
    "barge_in_ack": "बोलो जी।",
    "human_requested": "जरूर, म्हूँ माणस ने बुलावूँ।",
    "recording_consent": "कॉल रिकॉर्ड होवै सकै है।",
    "dtmf_fallback": "अंग्रेजी खातर एक अर हिंदी खातर दो दबावो, या दोबारा बोलो।",
}

# Marathi — the state language of Maharashtra, where the Dhule campus is.
# Dhule district (Khandesh) is multilingual: callers arrive in Marathi, Hindi,
# Ahirani and Gujarati, so Marathi is one of the three languages spoken in the
# opening prompt alongside English and Hindi.
MR = {
    "greeting": "एनएमआयएमएस ग्लोबल युनिव्हर्सिटी, धुळे येथे कॉल केल्याबद्दल धन्यवाद. आपण सारथी या एआय सहाय्यकाशी बोलत आहात.",
    "language_prompt": "कृपया आपली भाषा सांगा. Please tell me your preferred language.",
    "language_confirm": "ठीक आहे, मी आता {{language}} मध्ये बोलेन.",
    "language_unclear": "माफ करा, मला समजले नाही. कृपया आपल्या भाषेचे नाव सांगा, उदाहरणार्थ इंग्रजी, हिंदी किंवा मराठी.",
    "language_unsupported": "माफ करा, मला {{language}} येत नाही. मी इंग्रजी, हिंदी किंवा मराठीत मदत करू शकतो, किंवा एखाद्या व्यक्तीशी जोडू शकतो.",
    "menu": "आपण अभ्यासक्रम, प्रवेश, फी, पात्रता, शिष्यवृत्ती, वसतिगृह, प्लेसमेंट किंवा कॅम्पसबद्दल विचारू शकता. काय जाणून घ्यायचे आहे?",
    "no_input_1": "मी येथेच आहे. कृपया आपला प्रश्न विचारा.",
    "no_input_2": "मला काही ऐकू आले नाही. मी तुम्हाला प्रवेश विभागाशी जोडू का?",
    "thinking": "एक क्षण, मी तपासतो आहे.",
    "not_found": "मला ही माहिती सध्या उपलब्ध नाही. मी तुम्हाला एका व्यक्तीशी जोडतो.",
    "low_confidence": "मला याबद्दल पूर्ण खात्री नाही, म्हणून मी तुम्हाला प्रवेश विभागाशी जोडतो.",
    "escalate_offer": "तुम्हाला एखाद्या व्यक्तीशी बोलायचे आहे का?",
    "escalating": "आता तुम्हाला प्रवेश विभागाशी जोडत आहे. कृपया लाईनवर रहा.",
    "queue_hold": "आमचे सर्व सल्लागार व्यस्त आहेत. आपण रांगेत आहात, कृपया लाईनवर रहा.",
    "farewell": "एनएमआयएमएस ग्लोबल युनिव्हर्सिटी, धुळे येथे कॉल केल्याबद्दल धन्यवाद. नमस्कार.",
    "followup_offer": "ही माहिती मी तुम्हाला एसएमएस किंवा ईमेलने पाठवू का?",
    "barge_in_ack": "हो, बोला.",
    "human_requested": "नक्कीच, मी तुमच्यासाठी एका व्यक्तीला बोलवतो.",
    "recording_consent": "गुणवत्ता तपासणीसाठी ही कॉल रेकॉर्ड केली जाऊ शकते. नको असल्यास आम्हाला सांगा.",
    "dtmf_fallback": "सोपे असेल तर इंग्रजीसाठी एक, हिंदीसाठी दोन दाबा, किंवा पुन्हा बोलून सांगा.",
}

# Generic template for languages we support but have not fully scripted yet.
# Falls back to English prompts — the LLM still answers in the caller's language.
_GENERIC = EN


LANGUAGES: dict[str, Language] = {
    "en-IN": Language(
        code="en-IN",
        english_name="English",
        native_name="English",
        voices={
            "google": "en-IN-Neural2-B",
            "azure": "en-IN-NeerjaNeural",
            "elevenlabs": "Charlotte",
            "local": "en",
        },
        scripts=EN,
        markers=(
            "what", "the", "fee", "course", "admission", "please", "is", "for",
            "how", "can", "you", "and", "eligibility",
            # A caller's question is mostly function words and helpline nouns.
            # With only thirteen markers, ordinary English sentences such as
            # "When do classes start?" or "Are scholarships available?" scored no
            # evidence at all and came back with zero confidence, which re-prompts
            # a caller who has already told us, in English, what they want.
            "when", "where", "which", "who", "why", "are", "was", "were", "will",
            "would", "should", "could", "do", "does", "did", "have", "has", "had",
            "my", "your", "our", "this", "that", "these", "those", "there", "here",
            "want", "need", "tell", "know", "get", "give", "take", "come", "about",
            "from", "with", "not", "but", "or", "of", "to", "in", "on", "at", "by",
            "fees", "courses", "seats", "seat", "hostel", "scholarship",
            "scholarships", "placement", "placements", "documents", "document",
            "exam", "exams", "entrance", "apply", "application", "college",
            "university", "campus", "department", "programme", "program",
            "semester", "start", "starts", "last", "date", "available", "offer",
            "offers", "refund", "talk", "person", "help", "any", "many", "much",
        ),
    ),
    "hi-IN": Language(
        code="hi-IN",
        english_name="Hindi",
        native_name="हिन्दी",
        voices={
            "google": "hi-IN-Neural2-D",
            "azure": "hi-IN-MadhurNeural",
            "elevenlabs": "Aria",
            "local": "hi",
        },
        scripts=HI,
        markers=(
            "है", "और", "का", "की", "के", "मैं", "आप", "क्या", "को", "में", "करना",
            "हूँ", "बताइए", "कितनी", "फीस", "एडमिशन",
        ),
    ),
    "raj-IN": Language(
        # Rajasthani / Marwari. No mainstream ASR or TTS model exists for it, so:
        #  - LID: lexical detection on the transcript (Devanagari + Marwari markers)
        #  - ASR: recognised with the Hindi acoustic model (closest available)
        #  - TTS: spoken with a Hindi voice
        # This is a deliberate, documented trade-off — see docs/ARCHITECTURE.md.
        code="raj-IN",
        english_name="Rajasthani",
        native_name="राजस्थानी",
        asr_locale="hi-IN",
        tts_locale="hi-IN",
        lid_supported=False,
        fallback_code="hi-IN",
        voices={
            "google": "hi-IN-Neural2-D",
            "azure": "hi-IN-MadhurNeural",
            "elevenlabs": "Aria",
            "local": "hi",
        },
        scripts=RAJ,
        markers=(
            "म्हो", "म्हाणै", "म्हाड़ै", "थारूँ", "थानै", "कोनी", "आवै", "हूँ", "अठै",
            "रयो", "कै", "बातावो", "बण्या", "रहजो", "सा", "म्हूँ", "काई",
        ),
    ),
    "ta-IN": Language(
        code="ta-IN",
        english_name="Tamil",
        native_name="தமிழ்",
        voices={
            "google": "ta-IN-Neural2-A",
            "azure": "ta-IN-ValluvarNeural",
            "elevenlabs": "Aria",
            "local": "ta",
        },
        markers=("என்ன", "கட்டணம்", "சேர்க்கை", "நான்", "உங்கள்", "எப்படி", "குறித்து"),
    ),
    "bn-IN": Language(
        code="bn-IN",
        english_name="Bengali",
        native_name="বাংলা",
        voices={
            "google": "bn-IN-Neural2-A",
            "azure": "bn-IN-TanishaaNeural",
            "elevenlabs": "Aria",
            "local": "bn",
        },
        markers=("কি", "আমি", "আপনার", "ভর্তি", "ফি", "কত", "করতে"),
    ),
    "mr-IN": Language(
        code="mr-IN",
        english_name="Marathi",
        native_name="मराठी",
        voices={
            "google": "mr-IN-Neural2-A",
            "azure": "mr-IN-AarohiNeural",
            "elevenlabs": "Aria",
            "local": "mr",
        },
        markers=("आहे", "मी", "तुमची", "प्रवेश", "शुल्क", "किती", "करावे"),
        scripts=MR,
    ),
    "gu-IN": Language(
        code="gu-IN",
        english_name="Gujarati",
        native_name="ગુજરાતી",
        voices={
            "google": "gu-IN-Neural2-A",
            "azure": "gu-IN-DhwaniNeural",
            "elevenlabs": "Aria",
            "local": "gu",
        },
        markers=("છે", "હું", "તમારી", "પ્રવેશ", "ફી", "કેટલી", "કરવું"),
    ),
    "te-IN": Language(
        code="te-IN",
        english_name="Telugu",
        native_name="తెలుగు",
        voices={
            "google": "te-IN-Neural2-A",
            "azure": "te-IN-MohanNeural",
            "elevenlabs": "Aria",
            "local": "te",
        },
        markers=("ఏమిటి", "నేను", "మీ", "ప్రవేశం", "రుసుము", "ఎంత", "చేయాలి"),
    ),
    "kn-IN": Language(
        code="kn-IN",
        english_name="Kannada",
        native_name="ಕನ್ನಡ",
        voices={
            "google": "kn-IN-Neural2-A",
            "azure": "kn-IN-GaganNeural",
            "elevenlabs": "Aria",
            "local": "kn",
        },
        markers=("ಏನು", "ನಾನು", "ನಿಮ್ಮ", "ಪ್ರವೇಶ", "ಶುಲ್ಕ", "ಎಷ್ಟು", "ಮಾಡಬೇಕು"),
    ),
    "ml-IN": Language(
        code="ml-IN",
        english_name="Malayalam",
        native_name="മലയാളം",
        voices={
            "google": "ml-IN-Neural2-A",
            "azure": "ml-IN-SobhanaNeural",
            "elevenlabs": "Aria",
            "local": "ml",
        },
        markers=("എന്ത്", "ഞാൻ", "നിങ്ങളുടെ", "പ്രവേശനം", "ഫീസ്", "എത്ര", "ചെയ്യണം"),
    ),
    "pa-IN": Language(
        code="pa-IN",
        english_name="Punjabi",
        native_name="ਪੰਜਾਬੀ",
        voices={
            "google": "pa-IN-Neural2-A",
            "azure": "pa-IN-ArjunNeural",
            "elevenlabs": "Aria",
            "local": "pa",
        },
        markers=("ਕੀ", "ਮੈਂ", "ਤੁਹਾਡੀ", "ਦਾਖਲਾ", "ਫੀਸ", "ਕਿੰਨੀ", "ਕਰਨਾ"),
    ),
    "ur-IN": Language(
        code="ur-IN",
        english_name="Urdu",
        native_name="اردو",
        voices={
            "google": "ur-IN-Neural2-A",
            "azure": "ur-IN-SalmanNeural",
            "elevenlabs": "Aria",
            "local": "ur",
        },
        markers=("کیا", "میں", "آپ", "داخلہ", "فیس", "کتنی", "کرنا"),
    ),
    "or-IN": Language(
        code="or-IN",
        english_name="Odia",
        native_name="ଓଡ଼ିଆ",
        voices={"google": "or-IN-Neural2-A", "azure": "or-IN-SubhashiniNeural", "local": "or"},
        markers=("କଣ", "ମୁଁ", "ଆପଣଙ୍କ", "ପ୍ରବେଶ", "ଫି"),
    ),
    "as-IN": Language(
        code="as-IN",
        english_name="Assamese",
        native_name="অসমীয়া",
        voices={"google": "as-IN-Neural2-A", "azure": "as-IN-YashicaNeural", "local": "as"},
        markers=("কি", "মই", "আপোনাৰ", "ভৰ্তি"),
    ),
}

#: Unicode script -> candidate language codes (used by the lexical detector).
SCRIPT_BLOCKS: tuple[tuple[str, int, int, tuple[str, ...]], ...] = (
    ("Devanagari", 0x0900, 0x097F, ("hi-IN", "mr-IN", "raj-IN")),
    ("Bengali", 0x0980, 0x09FF, ("bn-IN", "as-IN")),
    ("Gurmukhi", 0x0A00, 0x0A7F, ("pa-IN",)),
    ("Gujarati", 0x0A80, 0x0AFF, ("gu-IN",)),
    ("Odia", 0x0B00, 0x0B7F, ("or-IN",)),
    ("Tamil", 0x0B80, 0x0BFF, ("ta-IN",)),
    ("Telugu", 0x0C00, 0x0C7F, ("te-IN",)),
    ("Kannada", 0x0C80, 0x0CFF, ("kn-IN",)),
    ("Malayalam", 0x0D00, 0x0D7F, ("ml-IN",)),
    ("Arabic", 0x0600, 0x06FF, ("ur-IN",)),
)

#: Romanised-Hindi / Hinglish markers — lets us detect Hindi spoken in Latin script.
HINGLISH_MARKERS = (
    "kya", "kyu", "kyun", "hai", "hain", "ho", "mujhe", "muje", "mera", "meri",
    "aap", "aapka", "aapki", "kitna", "kitni", "kitne", "kaun", "kaise", "kab",
    "kahan", "batao", "bataiye", "bataye", "chahiye", "karna", "karein", "kar",
    "padhai", "padhna", "lena", "dena", "hai?", "nahi", "acha", "accha", "theek",
    "thik", "saab", "ji", "bhai", "dijiye", "liye", "wala", "wali", "mein",
    "ka", "ki", "ke", "se", "par", "tak", "aur", "lekin", "agar", "toh",
)
# Bare "me" used to sit in this list. It is the Hindi postposition, but it is
# also one of the commonest English words, so "Tell me about placements." scored
# as romanised Hindi and was answered in Devanagari. "mein", "mujhe" and "mera"
# still carry the same signal without the collision.

#: Romanised Rajasthani/Marwari markers (subset that is unlikely in Hindi).
HINGLISH_RAJASTHANI_MARKERS = (
    "mhane", "mhare", "mharo", "kon", "koni", "aave", "aavai", "tharo", "thari",
    "thane", "thaaro", "bataavo", "batao", "atthe", "aithe", "ramram", "ram ram",
    "kai", "kad", "hove", "hoves", "banyo", "banya", "rai", "ryo",
)

#: Romanised-Marathi markers. Dhule callers very often say Marathi in Latin
#: script ("mala B.Tech chi mahiti pahije"), and without these the local LID
#: scored such a turn as English or Hinglish. Deliberately limited to words that
#: do not appear in romanised Hindi or English, so the marker score stays clean.
HINGLISH_MARATHI_MARKERS = (
    # pronouns / possessives
    "mala", "malhi", "mazi", "mazha", "mazhe", "majha", "majhe", "majhi",
    "tujha", "tujhi", "tumcha", "tumhala", "tumhi", "tumlhi", "apla", "apli",
    "aplyala", "tyacha", "tyachi", "hya",
    # copula / negation. "nahi" is deliberately absent: Hindi "नहीं" and Marathi
    # "नाही" both romanise to it, so it cannot separate the two.
    "aahe", "aahet", "aahot", "ahe", "nako",
    # interrogatives
    "kay", "kiti", "kasha", "kashi", "kase", "kunala", "kuthe", "kevha",
    "konta", "konti", "konte",
    # verb forms unique to Marathi
    "pahije", "hava", "havi", "havya", "karava", "karave", "karayache",
    "kara", "shikayache", "ghyayache", "ghyave", "bolu", "bolat", "bolato",
    "bolte", "sangaa", "sanga", "sangte", "mhanto", "mhantat", "mhanun",
    "det", "dyayche", "yet", "yete",
    # nouns / postpositions
    "madhe", "sathi", "vrutti", "mahiti", "abhyakram", "pravesh", "shulka",
    "vdyalay", "mahavidyalaya", "vasatigruh", "vishay", "varshi", "pudhe",
    "ata", "mag", "pan", "ani", "kiva", "mhanje",
)


def get_language(code: str | None) -> Language:
    """Look a language up, tolerating variants (`hi`, `hi_IN`, `HI-in`)."""
    if not code:
        return LANGUAGES["en-IN"]
    normalised = code.strip().replace("_", "-")
    if normalised in LANGUAGES:
        return LANGUAGES[normalised]
    lowered = normalised.lower()
    for key, lang in LANGUAGES.items():
        if key.lower() == lowered:
            return lang
    base = lowered.split("-")[0]
    for key, lang in LANGUAGES.items():
        if key.split("-")[0].lower() == base:
            return lang
    return LANGUAGES["en-IN"]


def asr_locale(code: str) -> str:
    lang = get_language(code)
    return lang.asr_locale or lang.code


def tts_locale(code: str) -> str:
    lang = get_language(code)
    return lang.tts_locale or lang.code


def voice_for(code: str, provider: str) -> str:
    lang = get_language(code)
    return lang.voices.get(provider, lang.voices.get("google", "en-IN-Neural2-B"))


def script_line(code: str, key: str, **kwargs: str) -> str:
    """Fetch a scripted prompt, filling `{{placeholders}}`."""
    lang = get_language(code)
    text = lang.scripts.get(key) or _GENERIC.get(key, "")
    for name, value in kwargs.items():
        text = text.replace("{{" + name + "}}", value)
    return text


def display_name(code: str, in_language: str | None = None) -> str:
    lang = get_language(code)
    if in_language and get_language(in_language).code != "en-IN":
        return lang.native_name
    return lang.english_name
