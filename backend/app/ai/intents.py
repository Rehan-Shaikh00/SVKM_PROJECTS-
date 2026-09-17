"""Intent detection.

Cheap, deterministic, multilingual (English + Devanagari Hindi + romanised
Hindi/Rajasthani). Used for three things:

1. filtering/boosting retrieval (a "fees" question should prefer fee records)
2. analytics (top queries, resolution rate per intent)
3. control flow (human_request → escalate, greeting → re-prompt the menu)

Deliberately rule-based: an LLM classifier would add 300–800 ms to every turn,
and the LLM still gets the final say on what the answer is.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

INTENTS = (
    "fees",
    "eligibility",
    "admission_process",
    "important_dates",
    "entrance_exam",
    "courses",
    "scholarships",
    "hostel",
    "placements",
    "facilities",
    "documents",
    "contact",
    "transport",
    "comparison",
    "loan_payment",
    "human_request",
    "greeting",
    "affirmation",
    "negation",
    "repeat",
    "smalltalk",
    "university_info",
    "other",
)

#: intent -> KB categories that answer it
INTENT_TO_CATEGORIES: dict[str, tuple[str, ...]] = {
    "fees": ("fees", "course", "loan_payment", "scholarships"),
    "eligibility": ("eligibility", "course", "admission_process"),
    "admission_process": ("admission_process", "important_dates", "documents", "eligibility"),
    "important_dates": ("important_dates", "admission_process"),
    "entrance_exam": ("entrance_exam", "admission_process", "eligibility"),
    "courses": ("course", "specialisation", "department"),
    "scholarships": ("scholarships", "fees"),
    "hostel": ("hostel", "facilities"),
    "placements": ("placements", "course"),
    "facilities": ("facilities", "hostel", "transport"),
    "documents": ("documents", "admission_process"),
    "contact": ("contact", "department"),
    "transport": ("transport", "facilities"),
    "comparison": ("course", "specialisation"),
    "loan_payment": ("loan_payment", "fees", "scholarships"),
    "university_info": ("university", "faq", "policy", "contact", "admission_process"),
}

_PATTERNS: dict[str, tuple[tuple[str, ...], float]] = {
    "fees": (
        (r"\bfees?\b", 1.0), (r"\bfee structure\b", 1.4), (r"\btuition\b", 1.2),
        (r"\bcost\b", 0.9), (r"\bcharges?\b", 0.8), (r"\bexpensiv", 0.6),
        (r"\bkitn[ai]\b", 0.9), (r"\bprice\b", 0.7), (r"\bper year\b", 0.8),
        (r"\btotal\b.*\b(pay|amount)\b", 0.8), (r"फीस", 1.3), (r"शुल्क", 1.2),
        # Marathi: "फी" is written with a single matra and "किती" asks the amount.
        (r"फी(?!स)", 1.2), (r"किती", 0.8), (r"खर्च", 0.9), (r"रक्कम", 1.0),
        (r"फ़ीस", 1.3), (r"kitni fees", 1.2), (r"khirch", 0.8), (r"ખર્ચ|રકમ", 0.8),
        (r"கட்டணம்", 1.2), (r"ফি", 1.1), (r"శుల్కం|ఫీజు", 1.2), (r"ಶುಲ್ಕ", 1.2),
        (r"ശുല്ക്കം|ഫീസ്", 1.2), (r"ਫੀਸ", 1.2), (r"فیس", 1.2), (r"ଟିକସ|ଫି", 1.0),
        (r"\brefund\b", 0.6), (r"\binstall?ment\b", 0.8), (r"\bemi\b", 0.8),
    ),
    "eligibility": (
        (r"\beligib", 1.4), (r"\bqualif", 1.1), (r"\brequire(ment|d)?\b", 1.0),
        (r"\bcriteri", 1.3), (r"\bcutoff\b|\bcut-?off\b|\bcut off\b", 1.2),
        (r"\bpercentage\b|\bmarks?\b", 0.9), (r"\bpaatrata\b|\bpatra\b", 1.1),
        (r"\bcan i (apply|get|join)\b", 1.3), (r"\bam i eligible\b", 1.4),
        (r"पात्रता", 1.4), (r"योग्यता", 1.3), (r"कितने पर्सेंट|कितने प्रतिशत", 1.2),
        # Marathi: "किती गुण आवश्यक आहेत", "टक्केवारी".
        (r"पात्र", 1.1), (r"गुण", 1.0), (r"टक्केवारी|टक्के", 1.2), (r"आवश्यक", 0.6),
        (r"मार्क्स", 0.9), (r"अंक", 0.8), (r"kabilyat", 1.0),
    ),
    "admission_process": (
        (r"\badmission\b|\badmissions\b|\bapply\b|\bapplication\b|\benrol", 1.2),
        (r"\bprocess\b|\bprocedure\b|\bsteps?\b", 0.8),
        (r"\bhow to (apply|get admission|register)\b", 1.4),
        (r"\bcounsel(l)?ing\b|\bmerit list\b|\bseat allot", 1.1),
        (r"प्रवेश", 1.3), (r"एडमिशन", 1.3), (r"आवेदन", 1.2), (r"दाखिला", 1.3),
        (r"एडमिशन कैसे", 1.4), (r"भरती", 0.9), (r"ప్రవేశం", 1.2), (r"ಪ್ರವೇಶ", 1.2),
    ),
    "important_dates": (
        (r"\bdeadline\b|\blast date\b|\bdue date\b|\bimportant dates?\b", 1.4),
        (r"\bwhen (do|does|will|is|are)\b.*\b(start|open|close|begin|end)\b", 1.2),
        (r"\bform\b.*\b(date|deadline)\b", 1.1), (r"\bsession\b|\bacademic year\b", 0.7),
        (r"\b20(2[5-9]|3\d)\b", 0.4),
        (r"अंतिम तिथि", 1.4), (r"कब तक", 1.2), (r"तारीख", 1.1), (r"आखिरी", 1.0),
        # Marathi: "शेवटची तारीख", "मुदत", "कधी". Weighted above "प्रवेश" so
        # "प्रवेशाची शेवटची तारीख" is a dates question, not a process question.
        (r"शेवटची तारीख|अंतिम तारीख", 1.8), (r"मुदत", 1.4), (r"डेडलाइन", 1.3),
        (r"कधी", 0.9),
        (r"आवेदन की अंतिम", 1.4),
        # Teaching-calendar questions: class start, term end exams, breaks. These
        # are weighted above entrance_exam's bare "परीक्षा" so "परीक्षा कब होगी?"
        # answers with dates rather than with the list of entrance tests.
        (r"\bacademic calendar\b|\bsemester (start|begin|dates?)\b|\bterm end exam|\bexam (dates?|schedule)\b|\bwhen (do|does) (the )?(class|classes|semester|term)", 1.6),
        (r"\bdiwali break\b|\bwinter vacation\b|\bsemester break\b|\bsports day\b|\bre-?exam\b", 1.3),
        (r"क्लास कब|कक्षा कब|सेमेस्टर कब|टर्म एंड|परीक्षा कब|परीक्षाएं कब|परीक्षा की तारीख|छुट्टी कब|शैक्षणिक कैलेंडर|दिनदर्शिका", 1.7),
        (r"वर्ग कधी|सेमेस्टर कधी|क्लास कधी|परीक्षा कधी|सुट्टी कधी|शैक्षणिक दिनदर्शिका|परीक्षेच्या तारखा|टर्म एंड परीक्षा", 1.8),
        # "Has the merit list come out?" and "which round is admission in?" are
        # timeline questions. Without these they fell to admission_process and the
        # caller was read the registration steps instead of the round status.
        (r"\bmerit list[s]?\b|\bwait(ing)? ?list\b|\bselection list\b", 1.5),
        (r"\bwhich round\b|\bwhat round\b|\bround (i|ii|iii|iv|1|2|3|4|one|two|three)\b|\b(first|second|third) round\b", 1.5),
        (r"\b(come|been) out\b|\bpublish(ed)?\b.*\b(list|result)\b|\badmission status\b|\bresult\b", 1.1),
        (r"मेरिट लिस्ट|गुणवत्ता सूची|चयन सूची|वेटिंग लिस्ट|कौन सा राउंड|राउंड|परिणाम आ", 1.5),
        (r"मेरिट यादी|गुणवत्ता यादी|निवड यादी|कोणती फेरी|फेरी|निकाल", 1.6),
        # "Is admission still open?" asks about the cycle, not the steps.
        (r"\bstill open\b|\badmission[s]? (is |are )?open\b|\bopen (for|till|until)\b|\bclosed\b|\bwindow\b", 1.4),
        (r"प्रवेश अभी खुला|अभी खुला है|आवेदन चालू|प्रवेश बंद", 1.5),
        (r"प्रवेश सुरू आहे|प्रवेश चालू आहे|प्रवेश बंद आहे|अर्ज सुरू", 1.6),
    ),
    "entrance_exam": (
        (r"\bentrance\b|\bexam\b|\btest\b", 1.1),
        (r"\b(npat|nmat|mht-?cet|cet|jee|clat|lsat|gate|gpat|cat|mat|xat|cmat|nata|"
        r"neet|next|nimsee)\b", 1.3),
        (r"\bsyllabus\b|\bpattern\b|\bprepar", 0.9),
        (r"प्रवेश परीक्षा", 1.4), (r"एंट्रेंस", 1.2), (r"परीक्षा", 1.1),
        (r"प्रवेश परीक्षा कोणती|कोणती परीक्षा", 1.5),
        # Hindi "प्रवेश के लिए कौन सी परीक्षा देनी होगी?" names the exam, not the
        # process: without these, "प्रवेश" (1.3) beat "परीक्षा" (1.1) and the caller
        # was read the admission steps instead of the qualifying tests.
        (r"कौन सी परीक्षा|कौनसी परीक्षा|परीक्षा देनी|किस परीक्षा", 1.6),
        (r"कोणत्या परीक्षेसाठी|परीक्षा द्यावी|कोणती परीक्षा द्यावी", 1.6),
    ),
    "courses": (
        (r"\b(courses?|programs?|programmes?|degrees?|streams?|branches?|speciali[sz]ations?)\b", 1.3),
        (r"\b(do you (have|offer)|is there|are there|available|offered)\b.*\b(course|program|programme|degree|branch)\b", 1.5),
        (r"\b(which|what)\b.*\b(courses?|programs?|programmes?|degrees?)\b", 1.3),
        (r"\blist of courses?\b|\ball courses\b|\bcourse catalogue\b", 1.4),
        # Intake questions belong here, not in fees. Marathi "किती जागा आहेत?" (how many seats?) used to score 0.8 on the
        # fees pattern for "किती" alone and the caller got the fee
        # escalation instead of the seat count.
        (r"\b(seats?|intake|capacity)\b", 1.2),
        (r"\bhow many seats\b|\bnumber of seats\b|\bseat matrix\b", 1.4),
        (r"जागा|बेठका|संख्या", 1.3), (r"सीटें|\bkitni seat", 1.3),
        (r"कोर्स", 1.3), (r"पाठ्यक्रम", 1.2), (r"विषय", 0.9), (r"ब्रांच", 1.0),
        # Marathi: "कोणते अभ्यासक्रम आहेत", "पदवी".
        (r"अभ्यासक्रम", 1.5), (r"कोणते कोर्स|कोणते अभ्यासक्रम", 1.6), (r"पदवी", 1.1),
        (r"कौन से कोर्स", 1.5), (r"कोर्सेस", 1.3),
        (r"కോర్సు", 1.2), (r"ಕೋರ್ಸ್", 1.2), (r"കോഴ്സ്", 1.2), (r"ਕੋਰਸ", 1.2),
    ),
    "university_info": (
        (r"\b(ugc|naac|aicte|nmc|dci|pci|inc|bci|coa|nba)\b", 1.4),
        (r"\b(recogni[sz]ed|recognition|accreditation|accredited|approved|valid|affiliated|affiliation|deemed|government|private)\b", 1.2),
        (r"\b(established|founded|ranking|rank|nirf|history|about (the )?university|who (are|is) you)\b", 1.1),
        (r"\b(nims|university|campus)\b.*\b(good|reputed|legit|genuine|fake)\b", 1.2),
        (r"मान्यता", 1.4), (r"मान्यता प्राप्त", 1.5), (r"यूजीसी", 1.4), (r"एकरेडिट", 1.2),
        (r"विश्वविद्यालय के बारे", 1.3), (r"प्राइवेट है या सरकारी", 1.4),
    ),
    "scholarships": (
        (r"\bscholarship\b|\bscholarships\b|\bconcession\b|\bwaiver\b|\bfreeship\b", 1.4),
        (r"\bfinancial aid\b|\bfee discount\b|\bmerit scholarship\b", 1.3),
        (r"छात्रवृत्ति", 1.4), (r"स्कॉलरशिप", 1.4), (r"स्काॅलरशिप", 1.4),
        # Marathi: "शिष्यवृत्ती", "फी माफी".
        (r"शिष्यवृत्ती", 1.6), (r"फी माफी|शुल्क माफी", 1.4), (r"अनुदान", 0.9),
        (r"छूट", 0.9), (r"ரஸ்காலர்ஷிப்", 1.2), (r"స్కాలర్‌షిప్", 1.2),
    ),
    "hostel": (
        (r"\bhostel\b|\bdorm\b|\broom\b|\baccommodation\b|\bwarden\b|\bmess\b", 1.3),
        (r"\bstay\b|\bliving\b|\bcampus residence\b", 0.8),
        (r"हॉस्टल", 1.4), (r"छात्रावास", 1.4), (r"कमरा", 0.8), (r"आवास", 1.0),
        # Marathi: "वसतिगृह", "हॉस्टेल" (long o), "राहण्याची सोय", "मेस".
        (r"वसतिगृह", 1.6), (r"हॉस्टेल", 1.5), (r"राहण्याची सोय", 1.4),
        # "मेस" must not match inside "सेमेस्टर" (semester): \b fails there because
        # Devanagari matras are not \w, so guard on the Devanagari block itself and
        # list the inflected forms a trailing guard would drop.
        (r"(?<![\u0900-\u097F])मेस(?![\u0900-\u097F])|मेसच[ीं]|मेसमध्ये", 1.0),
        (r"(?<![\u0900-\u097F])खोली", 0.8),
        # "हॉस्टल की सुविधा" also matches facilities ("सुविधा"), and facilities
        # used to win, sending a hostel question to the campus-facilities record.
        (r"हॉस्टल की सुविधा|हॉस्टेल ची सोय|वसतिगृह सुविधा|hostel ki suvidha", 2.2),
        (r"ಹಾಸ್ಟೆಲ್", 1.3), (r"హాస్టల్", 1.3),
    ),
    "placements": (
        (r"\bplacement\b|\bplacements\b|\bpackage\b|\bctc\b|\bsalary\b", 1.3),
        (r"\brecruiter\b|\bcompanies\b|\bhighest\b.*\b(package|ctc)\b", 1.2),
        (r"\bjob\b|\binternship\b|\bcampus drive\b|\bcompanies visit\b", 0.9),
        (r"प्लेसमेंट", 1.4), (r"नौकरी", 1.1), (r"वेतन", 1.0), (r"कंपनी", 0.8),
        (r"नोकरी", 1.2), (r"पगार", 1.0), (r"कंपन्या", 0.9),
    ),
    "facilities": (
        (r"\bfacilit|\blab\b|\blibrary\b|\bsports\b|\bgym\b|\bhospital\b", 1.2),
        (r"\bcampus\b|\binfrastructure\b|\bwi-?fi\b|\bclinic\b|\bamenit", 1.0),
        (r"सुविधा", 1.3), (r"लाइब्रेरी", 1.2), (r"लैब", 1.1), (r"कैंपस", 0.9),
        (r"ग्रंथालय", 1.2), (r"प्रयोगशाळा", 1.2), (r"कॅम्पस", 0.9),
        (r"हस्पताल|अस्पताल", 1.1),
    ),
    "documents": (
        (r"\bdocuments?\b|\bcertificates?\b|\bmarksheets?\b|\bmark sheets?\b|\bphotos?\b", 1.3),
        (r"\boriginals?\b|\baffidavit\b|\bproofs?\b|\bmigration\b|\btranscripts?\b", 1.1),
        (r"\b(what|which)\b.*\b(documents?|papers?|certificates?)\b", 1.5),
        (r"दस्तावेज", 1.3), (r"कागजात", 1.3), (r"प्रमाण पत्र", 1.1), (r"मार्कशीट", 1.2),
        # Marathi: "कागदपत्रे", "प्रमाणपत्र" (one word), "डॉक्युमेंट".
        (r"कागदपत्रे|कागदपत्र", 1.5), (r"प्रमाणपत्र", 1.2), (r"डॉक्युमेंट|डॉक्यूमेंट|डाक्यूमेंट|डॉक्युमेन्ट", 1.2),
    ),
    "contact": (
        # "what is the contact number for admissions" is a contact question even
        # though it mentions admissions, so the explicit phrase outweighs the
        # bare keyword that admission_process also scores on.
        (r"\b(?:contact|phone|helpline|enquiry|admission)\s+(?:number|no|details)\b", 1.9),
        (r"\btoll[\s-]?free\b|\bhelpline\b", 1.7),
        (r"\bcontact\b|\bphone\b|\bnumber\b|\bemail\b|\baddress\b", 1.2),
        # "how to reach" is transport, not contact: kept out on purpose.
        (r"\bwhere (is|are)\b|\blocation\b|\bpin code\b", 1.0),
        (r"\bdepartment\b|\bhod\b|\boffice\b", 0.9),
        (r"संपर्क क्रमांक|संपर्क नंबर|फोन क्रमांक|टोल फ्री|हेल्पलाइन", 1.9),
        (r"संपर्क", 1.3), (r"फोन नंबर", 1.3), (r"क्रमांक", 1.2),
        (r"पता|पत्ता", 1.1), (r"ईमेल", 1.1),
        (r"कहाँ|कहां", 0.8), (r"कैसै पहूँच", 1.3),
        (r"संपर्क करें|से संपर्क|संपर्क सूत्र", 1.5),
    ),
    "transport": (
        (r"\bbus\b|\btransport\b|\bshuttle\b|\bcommut", 1.3), (r"बस", 1.1),
        (r"परिवहन", 1.2),
        # Marathi: "कॅम्पसला कसे पोहोचावे", "वाहतूक", "रेल्वे", "विमानतळ".
        (r"\bhow (do i|can i|to) (reach|get to)\b", 1.6), (r"\breach the campus\b", 1.5),
        (r"\bdirections?\b", 1.3), (r"\bnearest (station|airport|railway)\b", 1.4),
        (r"कैसे पहुंचें|कैसे पहुँचें|कैसै पहूँच", 1.5),
        (r"कसे पोहोचावे|कसं पोहोचावं|कसे जावे", 1.6), (r"वाहतूक", 1.4),
        (r"रेल्वे|स्टेशन", 1.1), (r"विमानतळ", 1.2), (r"पत्ता", 1.0),
    ),
    "loan_payment": (
        (r"\bloan\b|\beducation loan\b|\bpayment\b|\bpay the fee\b|\bupi\b|\bneft\b", 1.2),
        (r"\brefund\b|\binstall", 0.9), (r"ऋण", 1.2), (r"लोन", 1.3), (r"भुगतान", 1.2),
        # Marathi and Hindi callers say "शिक्षण कर्ज" for an education loan,
        # "पैसे परत" for a refund and "प्रवेश रद्द" for withdrawing. None of those
        # were listed, so "शिक्षण कर्ज मिळेल का?" fell through to intent=other and
        # the caller was told which school runs a programme.
        (r"शिक्षण कर्ज|कर्ज|शैक्षणिक कर्ज", 1.4), (r"शिक्षण ऋण|ऋण मिलेगा|लोन मिलेगा", 1.4),
        (r"पैसे परत|पैसे वापस|रिफंड|फीस वापसी|शुल्क वापसी", 1.4),
        (r"प्रवेश रद्द|दाखिला रद्द|सीट छोड़", 1.3),
        (r"\b(withdraw|refund my|money back|cancel my admission)\b", 1.2),
    ),
    "comparison": (
        (r"\bvs\.?\b|\bversus\b|\bbetter\b|\bdifference between\b|\bcompare\b|\bwhich is\b", 1.2),
        (r"बेहतर", 1.1), (r"अंतर", 1.0), (r"कौन सा", 1.0),
    ),
    "human_request": (
        (r"\b(human|person|agent|representative|staff|someone|executive|officer|counsellor|counselor)\b", 1.0),
        (r"\b(talk|speak|connect|transfer)\s+(to|with)\b", 1.3),
        (r"\breal (person|human)\b", 1.4), (r"\bnot a (bot|robot|machine)\b", 1.4),
        (r"व्यक्ति से", 1.3), (r"इंसान", 1.4), (r"आदमी", 1.2), (r"बात करा", 1.3),
        (r"किसी से बात", 1.3), (r"माणस", 1.3), (r"माणस सूं", 1.4),
        (r"अधिकारी", 1.1), (r"customer care", 1.1), (r"complaint", 0.9), (r"शिकायत", 1.0),
    ),
    "greeting": (
        (r"^(hi|hello|hey|namaste|namaskar|good (morning|afternoon|evening))\b", 1.4),
        (r"^(नमस्ते|नमस्कार|हैलो|हेलो)", 1.4), (r"^(ram ram|राम राम|खम्मा घणी|खम्मा)", 1.4),
    ),
    "affirmation": (
        (r"^(yes|yeah|yep|ya|ok|okay|sure|fine|correct|right|please do|go ahead)\b", 1.4),
        (r"^(हाँ|हां|जी|ठीक|सही|बिल्कुल|जरूर|जी हाँ)", 1.4),
        (r"^(हो|ठीक आ|हांजी|जी हां)", 1.4),
    ),
    "negation": (
        (r"^(no|nope|nah|not|don'?t|dont|nothing|never)\b", 1.4),
        (r"^(नहीं|ना|मत|कुछ नहीं)", 1.4), (r"^(नाजी|कोनी|म्हाणे नाई)", 1.4),
    ),
    "repeat": (
        # Only fire on short, self-contained turns. A bare "what" inside a real
        # question ("what is the fee") must NOT be treated as a repeat request.
        (
            r"^\s*(repeat( that| it)?( please)?|again( please)?|say (that|it) again|"
            r"pardon( me)?|sorry\??|what\??|huh\??|come again( please)?|"
            r"excuse me\??|one more time)\s*[?.!]*\s*$",
            1.5,
        ),
        (
            r"\b(repeat that|say that again|say it again|could you repeat|"
            r"please repeat|didn'?t (hear|catch) (that|you)|"
            r"(could|can) not (hear|catch)|that was not clear)\b",
            1.2,
        ),
        (r"(दोबारा|फिर से|समझ नहीं आया|क्या कहा|दोहराइए|दोहराओ|एक बार फिर)", 1.3),
        (r"(फेर|दोबारा बोलो|फेर बोलो|समज नाई आलं)", 1.3),
    ),
    "smalltalk": (
        (r"\b(thank you|thanks|bye|goodbye|who are you|what are you|are you (a )?(bot|robot|ai|human))\b", 1.2),
        (r"(धन्यवाद|शुक्रिया|अलविदा|तुम कौन|आप कौन|तू कौन)", 1.3),
        (r"(थैंक यू|बाय|राम राम सा)", 1.2),
    ),
}

#: degree/course tokens we always try to pull out of the utterance
COURSE_TOKEN_RE = re.compile(
    r"\b(b\.?\s?tech|m\.?\s?tech|mbbs|bds|mds|b\.?\s?pharm|m\.?\s?pharm|pharm\.?\s?d|"
    r"d\.?\s?pharm|mba|bba|bca|mca|b\.?\s?sc|m\.?\s?sc|b\.?\s?a|m\.?\s?a|b\.?\s?com|"
    r"m\.?\s?com|b\.?\s?pt|m\.?\s?pt|b\.?\s?ot|b\.?\s?des|m\.?\s?des|b\.?\s?arch|"
    r"m\.?\s?arch|b\.?\s?plan|ll\.?\s?b|ll\.?\s?m|b\.?\s?a\.?\s?ll\.?\s?b|"
    r"b\.?\s?b\.?\s?a\.?\s?ll\.?\s?b|b\.?\s?ed|m\.?\s?ed|b\.?\s?p\.?\s?ed|"
    r"bhmct|b\.?\s?hm|bjmc|mjmc|bsw|msw|bpa|mpa|bva|mva|bfa|mfa|ph\.?\s?d|phd|"
    r"diploma|d\.?\s?el\.?\s?ed|gnm|anm|b\.?\s?voc|m\.?\s?voc)\b",
    re.IGNORECASE,
)

SPECIALISATION_RE = re.compile(
    r"\b(computer science|cse|it\b|information technology|artificial intelligence|"
    r"ai\b|machine learning|ml\b|data science|cyber security|cloud|iot|"
    r"mechanical|civil|electrical|electronics|ece|chemical|petroleum|"
    r"aerospace|robotics|automation|biotechnology|agriculture|nursing|"
    r"physiotherapy|pharmacology|pharmaceutics|finance|marketing|hr\b|human resource|"
    r"international business|business analytics|aviation|hotel management|"
    r"fashion|textile|interior|graphic|journalism|mass communication|law|"
    r"psychology|public health|nutrition|forensic|mathematics|physics|chemistry|"
    r"botany|zoology|english|hindi|history|political science|economics|education)\b",
    re.IGNORECASE,
)


#: Indian-script spellings of degree names -> canonical latin token.
#: Callers say "बीटेक" and ASR returns Devanagari; the latin regex cannot see it.
TRANSLIT_COURSE_MAP: dict[str, str] = {
    "बीटेक": "BTECH", "बी.टेक": "BTECH", "बी टेक": "BTECH", "बी. टेक": "BTECH",
    "एमटेक": "MTECH", "एम.टेक": "MTECH",
    "एमबीबीएस": "MBBS", "बीडीएस": "BDS", "एमडीएस": "MDS",
    "एमडी": "MD", "एमएस": "MS",
    "बीफार्म": "BPHARM", "बी.फार्म": "BPHARM", "एमफार्म": "MPHARM",
    "फार्मडी": "PHARMD", "डीफार्म": "DPHARM",
    "एमबीए": "MBA", "बीबीए": "BBA", "बीसीए": "BCA", "एमसीए": "MCA",
    "बीएससी": "BSC", "एमएससी": "MSC", "बीकॉम": "BCOM", "एमकॉम": "MCOM",
    "बीएड": "BED", "एमएड": "MED", "डीएलएड": "DELED",
    "एलएलबी": "LLB", "एलएलएम": "LLM",
    "बीआर्क": "BARCH", "एमआर्क": "MARCH",
    "जीएनएम": "GNM", "एएनएम": "ANM",
    "पीएचडी": "PHD", "डिप्लोमा": "DIPLOMA",
    "बीपीटी": "BPT", "एमपीटी": "MPT", "बीओटी": "BOT",
    "इंजीनियरिंग": "ENGINEERING", "नर्सिंग": "NURSING",
    "फिजियोथेरेपी": "BPT", "होटल मैनेजमेंट": "BHMCT", "पत्रकारिता": "BJMC",
}

#: Indian-script spellings of specialisations -> canonical latin label.
TRANSLIT_SPECIALISATION_MAP: dict[str, str] = {
    "कंप्यूटर साइंस": "computer science", "कंप्यूटर": "computer science",
    "सिविल": "civil", "मैकेनिकल": "mechanical", "इलेक्ट्रिकल": "electrical",
    "इलेक्ट्रॉनिक्स": "electronics", "केमिकल": "chemical", "पेट्रोलियम": "petroleum",
    "बायोटेक्नोलॉजी": "biotechnology", "बायोटेक": "biotechnology",
    "एआई": "artificial intelligence", "आर्टिफिशियल इंटेलिजेंस": "artificial intelligence",
    "फार्मेसी": "pharmacy", "कानून": "law", "मेडिकल": "medical", "डेंटल": "dental",
    "मैनेजमेंट": "management", "डिजाइन": "design", "फैशन": "fashion",
    "आर्किटेक्चर": "architecture", "एग्रीकल्चर": "agriculture", "कृषि": "agriculture",
    "पशु चिकित्सा": "veterinary", "होटल": "hotel management",
}


@dataclass
class IntentResult:
    intent: str = "other"
    confidence: float = 0.0
    scores: dict[str, float] = field(default_factory=dict)
    course_tokens: list[str] = field(default_factory=list)
    specialisations: list[str] = field(default_factory=list)
    numbers: list[str] = field(default_factory=list)
    #: candidate categories for retrieval boosting
    categories: tuple[str, ...] = ()
    text: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "intent": self.intent,
            "confidence": round(self.confidence, 3),
            "course_tokens": self.course_tokens,
            "specialisations": self.specialisations,
            "categories": list(self.categories),
        }


#: Intents that represent a *question about* a programme, as opposed to
#: "tell me about the programmes". Used to override a bare course-token match.
_QUESTION_INTENTS = frozenset({
    "fees", "eligibility", "admission_process", "important_dates", "entrance_exam",
    "scholarships", "hostel", "placements", "facilities", "documents", "contact",
    "transport", "comparison", "loan_payment", "university_info",
})
#: Minimum rival score needed to take over from an unsupported `courses` match.
_COURSE_TOKEN_OVERRIDE_MIN = 0.8


def _translit_hits(norm: str, mapping: dict[str, str]) -> set[str]:
    """Find non-overlapping matches, longest key first.

    Indian-script degree names collide as substrings of each other — `बीबीए`
    (BBA) sits inside `एमबीबीएस` (MBBS) — so a naive `key in norm` scan invents
    courses the caller never said. We walk the string once and, at each position,
    take the longest matching key and skip past it.
    """
    if not norm:
        return set()
    by_length = sorted(mapping, key=len, reverse=True)
    hits: set[str] = set()
    i, n = 0, len(norm)
    while i < n:
        for key in by_length:
            if key and norm.startswith(key, i):
                hits.add(mapping[key])
                i += len(key)
                break
        else:
            i += 1
    return hits


def normalise(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"[^\w\s.\-ऀ-ॿঀ-৿஀-௿ఀ-౿ಀ-೿ഀ-ൿ઀-૿਀-੿؀-ۿୀ-୿]", " ", text)
    return re.sub(r"\s+", " ", text)


def detect_intent(text: str) -> IntentResult:
    raw = text or ""
    norm = normalise(raw)
    scores: dict[str, float] = {}
    for intent, patterns in _PATTERNS.items():
        total = 0.0
        for pattern, weight in patterns:
            if re.search(pattern, norm):
                total += weight
        if total:
            scores[intent] = total

    # "courses" is inferred from degree tokens rather than keywords
    tokens = {t.replace(".", "").replace(" ", "").upper()
              for t in COURSE_TOKEN_RE.findall(norm)}
    tokens |= _translit_hits(norm, TRANSLIT_COURSE_MAP)
    course_tokens = sorted(tokens)

    specs = {s.strip().lower() for s in SPECIALISATION_RE.findall(norm)}
    specs |= _translit_hits(norm, TRANSLIT_SPECIALISATION_MAP)
    specialisations = sorted(specs)

    # How much of "courses" comes from an actual keyword ("which courses do you
    # offer") versus merely mentioning a degree token. Tracked separately below.
    courses_keyword_score = scores.get("courses", 0.0)
    if course_tokens or specialisations:
        scores["courses"] = courses_keyword_score + 1.1
    if len(course_tokens) >= 2 or ("which" in norm and "better" in norm):
        scores["comparison"] = scores.get("comparison", 0.0) + 0.6

    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    if not ranked:
        intent, top = "other", 0.0
    else:
        intent, top = ranked[0]

    # "what is the fee for B.Tech CSE" scores highest on `courses` only because
    # of the degree token; the caller is really asking about fees. When `courses`
    # has no keyword support of its own, let a genuine question intent win.
    if intent == "courses" and courses_keyword_score == 0.0:
        rivals = [(k, v) for k, v in scores.items()
                  if k in _QUESTION_INTENTS and v >= _COURSE_TOKEN_OVERRIDE_MIN]
        if rivals:
            intent, top = max(rivals, key=lambda kv: kv[1])
    total = sum(scores.values()) or 1.0
    confidence = min(0.99, 0.35 + 0.65 * (top / max(1.0, total)) * min(1.0, top / 1.5))

    numbers = re.findall(r"\d[\d,]*(?:\.\d+)?", norm)
    return IntentResult(
        intent=intent,
        confidence=round(confidence, 3),
        scores=scores,
        course_tokens=course_tokens,
        specialisations=specialisations,
        numbers=numbers,
        categories=INTENT_TO_CATEGORIES.get(intent, ()),
        text=raw,
    )


def is_control_utterance(text: str) -> tuple[bool, str]:
    """True for short conversational turns that need no retrieval."""
    result = detect_intent(text)
    if result.intent in {"human_request", "greeting", "affirmation", "negation",
                         "repeat", "smalltalk"} and result.confidence > 0.4:
        return True, result.intent
    return False, result.intent


def extract_entities(text: str) -> dict[str, list[str]]:
    norm = normalise(text)
    return {
        "course_tokens": sorted({t.replace(".", "").replace(" ", "").upper()
                                 for t in COURSE_TOKEN_RE.findall(norm)}),
        "specialisations": sorted({s.strip().lower() for s in SPECIALISATION_RE.findall(norm)}),
        "numbers": re.findall(r"\d[\d,]*(?:\.\d+)?", norm),
        "years": re.findall(r"\b20\d{2}\b", norm),
    }
