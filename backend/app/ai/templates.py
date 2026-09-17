"""Template answer composer — the zero-API-key grounded engine.

When `LLM_PROVIDER=local` (no Anthropic/OpenAI key) this module builds the spoken
answer directly from the retrieved records. It is deliberately **extractive and
structured**: every number it speaks comes from a `structured` field of a
retrieved record, or from a sentence copied out of the record body. It cannot
invent a fee, which is exactly the property we need on a helpline.

With a cloud LLM configured, the same retrieval result is handed to Claude
instead; this module still runs as the **fallback** if the LLM call fails or
times out, so the call never drops into silence.

Frames exist for English, Hindi, Marathi and Rajasthani — the languages a Dhule
helpline actually hears (Marathi first, then Hindi and English, with heavy
code-mixing). For any other language the English frame is used and
`fallback_language` is flagged, because the production path (Claude) generates
natively in that language.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from ..ai.spoken_numbers import (
    duration_phrase,
    spoken_count,
    spoken_money,
)
from ..kb.retriever import RetrievalResult, RetrievedChunk
from .intents import IntentResult

logger = logging.getLogger("nims.templates")

FULL_FRAMES = ("en-IN", "hi-IN", "mr-IN", "raj-IN")

#: Roughly 20 seconds of speech. Beyond this the caller loses the thread, so the
#: composer trims to sentence boundaries and offers the rest by message.
MAX_SPOKEN_CHARS = 320

FRAMES: dict[str, dict[str, str]] = {
    "en-IN": {
        "fees_both": "For {programme}, the fee is {annual} per year, so about {total} for the whole course.",
        "fees_annual": "For {programme}, the fee is {annual} per year.",
        "fees_total": "For {programme}, the total fee is {total}.",
        "fees_hostel": "Hostel is {hostel} per year on top of that.",
        "fees_unverified": "That is the last published figure, please confirm with admissions.",
        "fees_not_recorded": "I do not have the confirmed fee for {programme}. Let me connect you to the admissions office.",
        "fees_not_published": "The university does not publish a fee table on its website. The fee is confirmed at registration through the admission portal, or by the school office. Let me connect you to them.",
        "loan_not_published": "The university does not publish an education loan scheme, an interest rate or a procedure. A loan is the lender\u2019s decision, and the school office can issue the documents a bank asks for. Let me connect you to them.",
        "refund_not_published": "The university website does not publish a refund or withdrawal policy, so I cannot confirm a refund amount or a deadline. The accounts office handles that. Let me connect you to them.",
        "eligibility": "For {programme}, you need {eligibility}.",
        "eligibility_long": "For {programme}, the eligibility is: {eligibility}",
        "eligibility_follow": "You need {eligibility}.",
        "eligibility_exam": "Selection is through {exam}.",
        "entrance_tests_generic": "The qualifying test depends on the programme. For example, {examples}. Shall I send the list for every programme by SMS?",
        "specialisation_intake": "For {spec}, the intake is {seats} this cycle.",
        "duration": "{programme} runs for {duration}, with {seats} seats this cycle.",
        "duration_only": "{programme} is a {duration} full time programme.",
        "school": "It is offered by the {school}.",
        "admission_process": "The admission process is: {steps}.",
        "important_dates": "{dates}.",
        "scholarships": "We have {types}. {detail}",
        "hostel": "{summary}",
        "hostel_available": "Yes, hostel accommodation is available, with separate blocks for male and female students. Should I send the charges and how to apply by SMS?",
        "hostel_none": "Hostel accommodation is not available for this programme.",
        "hostel_unconfirmed": "The university website does not publish hostel details, so I cannot confirm accommodation or charges. Let me connect you to the school office.",
        "transport_summary": "The nearest airport is at {airport}. By road from Dhule it is {distances}.",
        "important_dates_mechanism": "Admission dates are published programme by programme and round by round on the university website. Should I send you the link by SMS?",
        "round_status": "For {programme}, {status}. Shall I send the schedule link by SMS?",
        "placements_not_published": "The university does not publish a placement report, package figures or a recruiter list, so I cannot confirm that. Let me connect you to the school office.",
        "campus_address": "The campus address is {address}.",
        "placements": "{summary}",
        "placement_numbers": "Last year the highest package was {highest} and the average was {average}.",
        "recruiters": "Recruiters include {names}.",
        "contact": "You can reach {department} on {phone}, or write to {email}.",
        "contact_phone": "The number is {phone}.",
        "contact_department_phone": "The {department} number is {phone}.",
        "contact_email": "The email is {email}.",
        "documents": "You will need {items}.",
        "facilities": "The campus has {items}.",
        "transport_rail": "{detail}",
        "transport_air": "{detail}",
        "transport_road": "By road from Dhule it is {distances}.",
        "transport": "{summary}",
        "catalog": "We offer {items}, and many more programmes.",
        "comparison": "Comparing the two: {left} is {left_fee}, and {right} is {right_fee}.",
        "not_found": "I don't have that information right now. Let me connect you to a human.",
        "offer_details": "I can send you the full list on WhatsApp or SMS, would that help?",
        "ask_clarify": "Which {programme} are you asking about?",
        "unverified_warning": "Please confirm this with the admissions office.",
        "generic": "{sentences}",
        "stale_warning": "That is from an earlier admission cycle.",
    },
    "hi-IN": {
        "fees_both": "{programme} की फीस {annual} प्रति वर्ष है, यानी पूरे कोर्स की लगभग {total}।",
        "fees_annual": "{programme} की फीस {annual} प्रति वर्ष है।",
        "fees_total": "{programme} की कुल फीस {total} है।",
        "fees_hostel": "इसके अलावा हॉस्टल की फीस {hostel} प्रति वर्ष है।",
        "fees_unverified": "यह पिछली प्रकाशित राशि है, कृपया एडमिशन ऑफिस से पुष्टि कर लें।",
        "fees_not_recorded": "{programme} की पुष्टि की गई फीस मेरे पास नहीं है। मैं आपको एडमिशन ऑफिस से जोड़ता हूँ।",
        "fees_not_published": "विश्वविद्यालय अपनी वेबसाइट पर शुल्क तालिका प्रकाशित नहीं करता। शुल्क की पुष्टि प्रवेश पोर्टल पर पंजीकरण के समय, या स्कूल कार्यालय से होती है। मैं आपको उनसे जोड़ता हूँ।",
        "loan_not_published": "विश्वविद्यालय शिक्षण ऋण की कोई योजना, ब्याज दर या प्रक्रिया प्रकाशित नहीं करता। ऋण बैंक का निर्णय है, और स्कूल कार्यालय बैंक के लिए आवश्यक दस्तावेज़ दे सकता है। मैं आपको उनसे जोड़ता हूँ।",
        "refund_not_published": "विश्वविद्यालय की वेबसाइट पर धनवापसी या प्रवेश रद्द करने की नीति प्रकाशित नहीं है, इसलिए मैं राशि या तारीख की पुष्टि नहीं कर सकता। यह काम अकाउंट्स कार्यालय देखता है। मैं आपको उनसे जोड़ता हूँ।",
        "eligibility": "{programme} के लिए {eligibility} चाहिए।",
        "eligibility_long": "{programme} के लिए पात्रता यह है: {eligibility}",
        "eligibility_follow": "इसके लिए {eligibility} चाहिए।",
        "eligibility_exam": "चयन {exam} के माध्यम से होता है।",
        "entrance_tests_generic": "प्रवेश परीक्षा पाठ्यक्रम पर निर्भर करती है। उदाहरण के लिए, {examples}। क्या मैं हर पाठ्यक्रम की सूची एसएमएस से भेज दूँ?",
        "specialisation_intake": "{spec} के लिए इस सत्र में {seats} सीटें हैं.",
        "duration": "{programme} की अवधि {duration} है और इस बार {seats} सीटें हैं।",
        "duration_only": "{programme} एक {duration} का पूर्णकालिक कोर्स है।",
        "school": "यह कोर्स {school} द्वारा संचालित है।",
        "admission_process": "एडमिशन की प्रक्रिया है: {steps}।",
        "important_dates": "{dates}।",
        "scholarships": "हमारे पास {types} हैं। {detail}",
        "hostel": "{summary}",
        "hostel_available": "जी हाँ, वसतिगृह की सुविधा उपलब्ध है, लड़के और लड़कियों के लिए अलग अलग ब्लॉक हैं। क्या मैं शुल्क और आवेदन की जानकारी एसएमएस से भेज दूँ?",
        "hostel_none": "इस कोर्स के लिए वसतिगृह की सुविधा उपलब्ध नहीं है।",
        "hostel_unconfirmed": "विश्वविद्यालय की वेबसाइट पर हॉस्टल की जानकारी प्रकाशित नहीं है, इसलिए मैं आवास या शुल्क की पुष्टि नहीं कर सकता। मैं आपको स्कूल कार्यालय से जोड़ता हूँ।",
        "transport_summary": "सबसे नज़दीकी हवाई अड्डा {airport} है। धुले से सड़क मार्ग से दूरी है {distances}।",
        "important_dates_mechanism": "प्रवेश की तारीखें विश्वविद्यालय की वेबसाइट पर कोर्स और राउंड के हिसाब से प्रकाशित की जाती हैं। क्या मैं लिंक एसएमएस से भेज दूँ?",
        "round_status": "{programme} के लिए, {status}। क्या मैं शेड्यूल का लिंक एसएमएस से भेज दूँ?",
        "placements_not_published": "विश्वविद्यालय प्लेसमेंट रिपोर्ट, पैकेज के आंकड़े या रिक्रूटर सूची प्रकाशित नहीं करता, इसलिए मैं इसकी पुष्टि नहीं कर सकता। मैं आपको स्कूल कार्यालय से जोड़ता हूँ।",
        "campus_address": "कैंपस का पता है {address}।",
        "placements": "{summary}",
        "placement_numbers": "पिछले साल सबसे ऊँचा पैकेज {highest} रहा और औसत {average}।",
        "recruiters": "कंपनियों में {names} शामिल हैं।",
        "contact": "{department} से आप {phone} पर संपर्क कर सकते हैं, या {email} पर लिख सकते हैं।",
        "contact_phone": "नंबर है {phone}।",
        "contact_department_phone": "{department} का नंबर है {phone}।",
        "contact_email": "ईमेल है {email}।",
        "documents": "आपको {items} चाहिए होंगे।",
        "facilities": "कैंपस में {items} हैं।",
        "transport_rail": "ट्रेन से यात्रा: {detail}",
        "transport_air": "हवाई यात्रा: {detail}",
        "transport_road": "धुले से सड़क मार्ग से दूरी है {distances}।",
        "transport": "{summary}",
        "catalog": "हमारे पास {items} और भी कई कोर्स हैं।",
        "comparison": "दोनों की तुलना में: {left} की फीस {left_fee} है, और {right} की {right_fee}।",
        "not_found": "मुझे यह जानकारी अभी उपलब्ध नहीं है। मैं आपको किसी व्यक्ति से जोड़ता हूँ।",
        "offer_details": "मैं पूरी सूची व्हाट्सएप या एसएमएस पर भेज सकता हूँ, भेजूँ?",
        "ask_clarify": "आप किस {programme} के बारे में पूछ रहे हैं?",
        "unverified_warning": "कृपया इसे एडमिशन ऑफिस से पुष्टि कर लें।",
        "generic": "{sentences}",
        "stale_warning": "यह जानकारी पिछले एडमिशन सत्र की है।",
    },
    "mr-IN": {
        "fees_both": "{programme} चे शुल्क प्रति वर्ष {annual} आहे, म्हणजे संपूर्ण अभ्यासक्रमासाठी सुमारे {total}.",
        "fees_annual": "{programme} चे शुल्क प्रति वर्ष {annual} आहे.",
        "fees_total": "{programme} चे एकूण शुल्क {total} आहे.",
        "fees_hostel": "याव्यतिरिक्त वसतिगृहाचे शुल्क प्रति वर्ष {hostel} आहे.",
        "fees_unverified": "ही अंतिम प्रसिद्ध झालेली रक्कम आहे, कृपया प्रवेश कार्यालयाकडून खात्री करून घ्या.",
        "fees_not_recorded": "{programme} ची अधिकृत फी माझ्याकडे नोंदलेली नाही. मी तुम्हाला प्रवेश कार्यालयाशी जोडतो.",
        "fees_not_published": "विद्यापीठ आपल्या संकेतस्थळावर शुल्क तालिका प्रसिद्ध करत नाही. शुल्काची पुष्टी प्रवेश पोर्टलवर नोंदणीच्या वेळी, किंवा शाळेच्या कार्यालयाकडून होते. मी तुम्हाला त्यांच्याशी जोडतो.",
        "loan_not_published": "विद्यापीठ शिक्षण कर्ज योजना, व्याजदर किंवा प्रक्रिया प्रसिद्ध करत नाही. कर्ज हा बँकेचा निर्णय असतो आणि शाळेचे कार्यालय बँकेला आवश्यक असलेली कागदपत्रे देऊ शकते. मी तुम्हाला त्यांच्याशी जोडतो.",
        "refund_not_published": "विद्यापीठाच्या संकेतस्थळावर परतावा किंवा प्रवेश रद्द करण्याची नीती प्रसिद्ध नाही, त्यामुळे मी रक्कम किंवा मुदत सांगू शकत नाही. हे काम लेखा कार्यालय पाहते. मी तुम्हाला त्यांच्याशी जोडतो.",
        "eligibility": "{programme} साठी {eligibility} आवश्यक आहे.",
        "eligibility_long": "{programme} साठी पात्रता अशी आहे: {eligibility}",
        "eligibility_follow": "यासाठी {eligibility} आवश्यक आहे.",
        "eligibility_exam": "निवड {exam} द्वारे होते.",
        "entrance_tests_generic": "प्रवेश परीक्षा अभ्यासक्रमावर अवलंबून असते. उदाहरणार्थ, {examples}. प्रत्येक अभ्यासक्रमाची यादी एसएमएसने पाठवू का?",
        "specialisation_intake": "{spec} साठी या वर्षी {seats} जागा आहेत.",
        "duration": "{programme} चा कालावधी {duration} आहे आणि या वर्षी {seats} जागा आहेत.",
        "duration_only": "{programme} हा {duration} कालावधीचा पूर्णवेळ अभ्यासक्रम आहे.",
        "school": "हा अभ्यासक्रम {school} द्वारे चालवला जातो.",
        "admission_process": "प्रवेश प्रक्रिया अशी आहे: {steps}.",
        "important_dates": "{dates}.",
        "scholarships": "आमच्याकडे {types} आहेत. {detail}",
        "hostel": "{summary}",
        "hostel_available": "होय, वसतिगृह उपलब्ध आहे, मुले आणि मुलींसाठी स्वतंत्र ब्लॉक आहेत. शुल्क आणि अर्ज प्रक्रियेची माहिती एसएमएसने पाठवू का?",
        "hostel_none": "या अभ्यासक्रमासाठी वसतिगृह उपलब्ध नाही.",
        "hostel_unconfirmed": "विद्यापीठाच्या संकेतस्थळावर वसतिगृहाची माहिती प्रसिद्ध केलेली नाही, त्यामुळे मी राहण्याची सोय किंवा शुल्क सांगू शकत नाही. मी तुम्हाला शाळेच्या कार्यालयाशी जोडतो.",
        "transport_summary": "सर्वात जवळचे विमानतळ {airport} आहे. धुळ्यापासून रस्त्याने अंतरे आहेत {distances}.",
        "important_dates_mechanism": "प्रवेशाच्या तारखा विद्यापीठाच्या संकेतस्थळावर अभ्यासक्रम आणि फेरीनुसार प्रसिद्ध केल्या जातात. लिंक एसएमएसने पाठवू का?",
        "round_status": "{programme} साठी, {status}. वेळापत्रकाची लिंक एसएमएसने पाठवू का?",
        "placements_not_published": "विद्यापीठ प्लेसमेंट अहवाल, पॅकेज आकडे किंवा रिक्रुटर यादी प्रसिद्ध करत नाही, त्यामुळे मी याची पुष्टी करू शकत नाही. मी तुम्हाला शाळेच्या कार्यालयाशी जोडतो.",
        "campus_address": "कॅम्पसचा पत्ता आहे {address}.",
        "placements": "{summary}",
        "placement_numbers": "गेल्या वर्षी सर्वाधिक पॅकेज {highest} होते आणि सरासरी {average} होती.",
        "recruiters": "कंपन्यांमध्ये {names} यांचा समावेश आहे.",
        "contact": "{department} शी तुम्ही {phone} वर संपर्क साधू शकता, किंवा {email} वर लिहू शकता.",
        "contact_phone": "क्रमांक आहे {phone}.",
        "contact_department_phone": "{department}चा नंबर आहे {phone}.",
        "contact_email": "ईमेल आहे {email}.",
        "documents": "तुम्हाला {items} आवश्यक असतील.",
        "facilities": "कॅम्पसमध्ये {items} आहेत.",
        "transport_rail": "रेल्वेने प्रवास: {detail}",
        "transport_air": "विमानाने प्रवास: {detail}",
        "transport_road": "धुळ्यापासून रस्त्याने अंतरे आहेत {distances}.",
        "transport": "{summary}",
        "catalog": "आमच्याकडे {items} आणि अजून अनेक अभ्यासक्रम आहेत.",
        "comparison": "दोघांची तुलना करता: {left} चे शुल्क {left_fee} आहे, आणि {right} चे {right_fee}.",
        "not_found": "मला ही माहिती सध्या उपलब्ध नाही. मी तुम्हाला एका व्यक्तीशी जोडतो.",
        "offer_details": "मी पूर्ण यादी व्हॉट्सॲप किंवा एसएमएसने पाठवू शकतो, पाठवू का?",
        # Marathi inflects the noun before a postposition, so the slot carries
        # the oblique stem and "विषयी" is joined to it: "कोणत्या अभ्यासक्रमाविषयी",
        # not "कोणत्या अभ्यासक्रम विषयी".
        "ask_clarify": "तुम्ही कोणत्या {programme}विषयी विचारत आहात?",
        "unverified_warning": "कृपया प्रवेश कार्यालयाकडून खात्री करून घ्या.",
        "generic": "{sentences}",
        "stale_warning": "ही माहिती मागील प्रवेश सत्रातील आहे.",
    },
    "raj-IN": {
        "fees_both": "{programme} री फीस {annual} प्रति वर्ष है, यानी पूरे कोर्स री लगभग {total}।",
        "fees_annual": "{programme} री फीस {annual} प्रति वर्ष है।",
        "fees_total": "{programme} री कूल फीस {total} है।",
        "fees_hostel": "इसका अलावा हॉस्टल री फीस {hostel} प्रति वर्ष है।",
        "fees_unverified": "ई पिछली छपी राशि है, एडमिशन ऑफिस सूं पक्की कर लेवजो।",
        "fees_not_recorded": "{programme} री पक्की फीस म्हारै पास कोनी है। म्हूँ थानै एडमिशन ऑफिस सूं जोड़ दूँ।",
        "fees_not_published": "युनिवर्सिटी आप री वेबसाइट पर शुल्क री तालिका छाप कोनी है। शुल्क री बात प्रवेश पोर्टल पर रजिस्ट्रेशन का वखत, या स्कूल ऑफिस सूं पक्की होवै है। म्हूँ थानै ऊँ सूं जोड़ दूँ।",
        "loan_not_published": "युनिवर्सिटी शिक्षण ऋण री कोई योजना, ब्याज दर या प्रक्रिया छापती नांई। ऋण बैंक री मर्जी है, अर स्कूल ऑफिस बैंक नै लागता कागद दे सकै है। म्हूँ थानै ऊँ सूं जोड़ दूँ।",
        "refund_not_published": "युनिवर्सिटी री वेबसाइट पर पैसा वापसी या दाखिलो रद्द करण री नीति छपी नांई, इसर म्हूँ रकम या तारख नी कैं सकूँ। ई काम ऑफिस देखै है। म्हूँ थानै ऊँ सूं जोड़ दूँ।",
        "eligibility": "{programme} खातर {eligibility} चाइए।",
        "eligibility_long": "{programme} खातर पात्रता ई है: {eligibility}",
        "eligibility_follow": "ई खातर {eligibility} चाइए।",
        "eligibility_exam": "चयण {exam} सूं होवै है।",
        "entrance_tests_generic": "प्रवेश परीक्षा कोर्स पर टिकै है। मसलन, {examples}। म्हूँ हर कोर्स री लिस्ट एसएमएस सूं भेज दूँ?",
        "specialisation_intake": "{spec} खातर ई साल {seats} सीटां है.",
        "duration": "{programme} री अवधि {duration} है अर ई बार {seats} सीटां है।",
        "duration_only": "{programme} एक {duration} रो फुल टाइम कोर्स है।",
        "school": "ई कोर्स {school} चलावै है।",
        "admission_process": "एडमिशन री प्रक्रिया है: {steps}।",
        "important_dates": "{dates}।",
        "scholarships": "म्हाड़ै {types} है। {detail}",
        "hostel": "{summary}",
        "hostel_available": "हाँजी, हॉस्टेल री सुविधा है, छात्र अर छात्रावां खातर अलग अलग ब्लॉक है। शुल्क अर आवेदन री बात एसएमएस सूं भेज दूँ?",
        "hostel_none": "ई कोर्स खातर हॉस्टेल री सुविधा कोनी है।",
        "hostel_unconfirmed": "युनिवर्सिटी री वेबसाइट पर हॉस्टेल री बात छपी कोनी है, इसलिये म्हूँ रहण सूं या शुल्क री बात पक्की कोनी कर सको। म्हूँ थानै स्कूल ऑफिस सूं जोड़ दूँ।",
        "transport_summary": "सब सूं नीड़ो हवाई अड्डो {airport} है। धुले सूं सड़क रास्ते दूरी है {distances}।",
        "important_dates_mechanism": "प्रवेश री तारीखां युनिवर्सिटी री वेबसाइट पर कोर्स अर राउंड का हिसाब सूं छपै है। एसएमएस सूं लिंक भेज दूँ?",
        "round_status": "{programme} खातर, {status}। म्हूँ शेड्यूल री लिंक एसएमएस सूं भेज दूँ?",
        "placements_not_published": "युनिवर्सिटी प्लेसमेंट रिपोर्ट, पैकेज रा आँकडा या रिक्रूटर सूची छपै नहीं, इसलियै म्हूँ ए की पुष्टि कोनी कर सकूँ। म्हूँ थानै स्कूल ऑफिस सूं जोड़ दूँ।",
        "campus_address": "कैंपस रो पतो है {address}।",
        "placements": "{summary}",
        "placement_numbers": "गत साल सारूँ ऊँचो पैकेज {highest} रह्यो अर औसत {average}।",
        "recruiters": "कंपनियों में {names} शामिल है।",
        "contact": "{department} सूं थूँ {phone} पर बात कर सको, या {email} पर लिख सको।",
        "contact_phone": "नंबर है {phone}।",
        "contact_department_phone": "{department} रो नंबर है {phone}।",
        "contact_email": "ईमेल है {email}।",
        "documents": "थानै {items} चाइए पड़सी।",
        "facilities": "कैंपस में {items} है।",
        "transport_rail": "ट्रेन सूं यात्रा: {detail}",
        "transport_air": "हवाई यात्रा: {detail}",
        "transport_road": "धुले सूं सड़क रास्ते दूरी है {distances}।",
        "transport": "{summary}",
        "catalog": "म्हाड़ै {items} अर भी ढेर कोर्स है।",
        "comparison": "दूवां री तुलना में: {left} री फीस {left_fee} है, अर {right} री {right_fee}।",
        "not_found": "म्हाड़ै ई जाणकारी कोनी है। म्हूँ थानै किसी माणस सूं मिला दूँ।",
        "offer_details": "म्हूँ पूरी लिस्ट वाट्सएप या एसएमएस सूं भेज सकूँ, भेजूँ?",
        "ask_clarify": "थूँ किस {programme} बारे पूछ रया हौ?",
        "unverified_warning": "एडमिशन ऑफिस सूं पक्की कर लेवजो।",
        "generic": "{sentences}",
        "stale_warning": "ई जाणकारी गत सत्र री है।",
    },
}


@dataclass
class ComposedAnswer:
    text: str
    confidence: float = 0.0
    grounded: bool = False
    needs_escalation: bool = False
    escalation_reason: str | None = None
    followup: dict[str, Any] | None = None
    intent: str = "other"
    language: str = "en-IN"
    fallback_language: bool = False
    citations: list[dict[str, Any]] = field(default_factory=list)
    template: str = ""
    debug: dict[str, Any] = field(default_factory=dict)


# Spoken when a contact record names no department of its own, so the frame
# never says "You can reach None on …".
DEPARTMENT_FALLBACK = {
    "en": "the admissions office",
    "hi": "प्रवेश कार्यालय",
    "mr": "प्रवेश कार्यालय",
    "raj": "प्रवेश कार्यालय",
}


#: Categories whose record title is the name of a programme.
PROGRAMME_CATEGORIES = frozenset({"course", "specialisation"})

#: What to call the subject of a question when the record that answered it is a
#: policy or FAQ record rather than a programme record.
#: What `ask_clarify` puts in its slot. It cannot take PROGRAMME_NEUTRAL: "Which
#: this programme are you asking about?" is not a sentence.
PROGRAMME_NOUN = {
    "en": "programme",
    "hi": "\u092a\u093e\u0920\u094d\u092f\u0915\u094d\u0930\u092e",
    "mr": "\u0905\u092d\u094d\u092f\u093e\u0938\u0915\u094d\u0930\u092e\u093e",
    "raj": "\u0915\u094b\u0930\u094d\u0938",
}

PROGRAMME_NEUTRAL = {
    "en": "this programme",
    "hi": "\u0907\u0938 \u092a\u093e\u0920\u094d\u092f\u0915\u094d\u0930\u092e",
    "mr": "\u0939\u093e \u0905\u092d\u094d\u092f\u093e\u0938\u0915\u094d\u0930\u092e",
    "raj": "\u0908 \u0915\u094b\u0930\u094d\u0938",
}


def _programme_named(intent: Any) -> bool:
    """Did the caller say which programme they mean?

    True when this turn named one, and also when a previous turn did and the
    engine carried it in — by the time compose runs, both are in `course_tokens`.
    """
    return bool(getattr(intent, "course_tokens", None) or getattr(intent, "specialisations", None))


def _frames(language: str) -> dict[str, str]:
    if language in FRAMES:
        return FRAMES[language]
    base = language.split("-")[0]
    for code, frame in FRAMES.items():
        if code.split("-")[0] == base:
            return frame
    return FRAMES["en-IN"]


def _money(value: Any, language: str, per: str | None = None) -> str:
    return spoken_money(value, language, per=per)


def _merge_structured(items: list[RetrievedChunk]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for item in items:
        for key, value in (item.structured or {}).items():
            if key == "fees" and isinstance(value, dict):
                fees = merged.setdefault("fees", {})
                for fee_key, fee_value in value.items():
                    fees.setdefault(fee_key, fee_value)
            else:
                merged.setdefault(key, value)
    return merged


def _group_by_record(items: list[RetrievedChunk]) -> list[tuple[RetrievedChunk, list[RetrievedChunk]]]:
    grouped: dict[str, list[RetrievedChunk]] = {}
    order: list[str] = []
    for item in items:
        if item.record_id not in grouped:
            grouped[item.record_id] = []
            order.append(item.record_id)
        grouped[item.record_id].append(item)
    return [(grouped[rid][0], grouped[rid]) for rid in order]


# A caller who asks "how do I reach the campus by train" must not be told about
# the airport. The record publishes by_rail / by_air / road_distances separately,
# so answer the mode the question actually names.
_RAIL_RE = re.compile(
    r"\btrains?\b|\brailways?\b|\brail\b|ट्रेन|रेलवे|रेल्वे|रेलगाडी|रेल",
    re.IGNORECASE,
)
_AIR_RE = re.compile(
    r"\bflights?\b|\bplanes?\b|\bairports?\b|\bair\b|\bfly\b|\bflying\b|विमान|हवाई|फ्लाइट|विमानतळ",
    re.IGNORECASE,
)
_ROAD_RE = re.compile(
    r"\bbus\b|\bbuses?\b|\bcars?\b|\bdriv\w*\b|\broad\b|\btaxi\b|\bcab\b"
    # Hindi "गाड़ी/सड़क/रस्ता", Marathi "गाडी/रस्त्याने/एसटी" (the short-a forms the
    # Hindi spellings do not cover).
    r"|बस|गाड़ी|गाडी|सड़क|रस्ता|रस्त्याने|मार्ग|महामार्ग|एसटी",
    re.IGNORECASE,
)


_BARE_DEGREE_RE = re.compile(
    r"\b(b\.?tech|b\.?e|b\.?pharm|b\.?pharmacy|d\.?pharm|d\.?pharmacy|m\.?pharm"
    r"|m\.?tech|mca|bba|bca|b\.?sc|m\.?sc|mba|ph\.?d)\b",
    re.IGNORECASE,
)
# Words a fee question is made of, in the languages this assistant serves. They
# carry no programme information, so they must not count as a specialisation.
_QUESTION_FILLER = {
    "what", "whats", "is", "are", "the", "a", "an", "for", "of", "to", "in", "at",
    "fee", "fees", "cost", "charges", "charge", "total", "annual", "yearly",
    "per", "year", "how", "much", "many", "kitna", "kitni", "kitne", "kya", "hai",
    "hain", "ki", "ka", "ke", "kar", "karke", "dena", "batana", "bataye", "batao",
    "please", "tell", "me", "i", "do", "you", "have", "and", "or", "with",
    "किती", "आहे", "काय", "शुल्क", "फी", "किंमत", "सांगा", "वर्ष", "साठी",
    "que", "quanto",
}


def _caller_named_only_the_degree(question: str, programme: str) -> str | None:
    """The degree prefix to speak, when the caller named nothing more specific.

    "What is the fee for B.Tech?" retrieved B.Tech (Cosmetic Technology), so the
    caller was told "I do not have the confirmed fee for B.Tech (Cosmetic
    Technology)" — a programme nobody asked about, when seven other B.Tech records
    exist. Since no fee is published for any of them, the honest answer names the
    degree the caller actually said. Returns None when the caller did name a
    specialisation that the winning record matches.
    """
    match = _BARE_DEGREE_RE.search(question or "")
    if not match:
        return None
    prefix = match.group(0)
    prefix_core = prefix.lower().replace(".", "")
    title = (programme or "").lower()
    # Only narrow a *variant* title; a record titled exactly the prefix is already right.
    if prefix_core not in title.replace(".", "") or title.strip() == prefix_core:
        return None
    title_words = set(re.findall(r"[a-z]{3,}", title)) - {prefix_core, "tech", "pharm"}
    asked_words = {
        w for w in re.findall(r"[a-z]{3,}", (question or "").lower())
        if w not in _QUESTION_FILLER and w != prefix_core
    }
    # The caller used a word that belongs to this record's own name: they were
    # specific, so keep the record's title.
    if asked_words & title_words:
        return None
    return prefix


# "What is the highest package?" asks for a placement detail the university has
# never published; "do you have placements?" does not, and the homepage claim is a
# fair answer to that one.
_PLACEMENT_DETAIL_RE = re.compile(
    r"\bpackages?\b|\bctc\b|\bsalar(?:y|ies)\b|\bhighest\b|\baverage\b|\bmedian\b"
    r"|\brecruiters?\b|\bcompanies\b|\bplacement report\b|\bplacement percentage\b"
    r"|पैकेज|सैलरी|वेतन|पगार|रिक्रूटर|कंपनी|कंपनियाँ|प्लेसमेंट कितनी"
    r"|पॅकेज|रिक्रुटर|कंपन्या|नोकरी किती|प्लेसमेंट किती",
    re.IGNORECASE,
)

# An address question. The contact record also carries phone numbers, so without
# this a caller asking for the address was read a number or the landmarks.
_ADDRESS_RE = re.compile(
    r"\baddress\b|\blocation\b|\bwhere (is|are)\b|\bdirections?\b|\bpin ?code\b"
    r"|पत्ता|पता|ठिकाण|कोठे|कुठे|कहाँ है|कहां है|पिन कोड",
    re.IGNORECASE,
)


def _transport_mode(text: str) -> str | None:
    """The mode of travel a caller named, or None for a general 'how to reach'."""
    sample = text or ""
    if _RAIL_RE.search(sample):
        return "rail"
    if _AIR_RE.search(sample):
        return "air"
    if _ROAD_RE.search(sample):
        return "road"
    return None


def _join(items: list[str], language: str) -> str:
    """Join short list items with a language-appropriate conjunction.

    Trailing stops are stripped first: items come from knowledge-base lists that
    are sometimes written as sentences, and "A. and B." reads badly aloud.
    """
    cleaned = [i.strip().rstrip(".").strip() for i in items if i and i.strip()]
    cleaned = [c for c in cleaned if c]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    base = language.split("-")[0]
    if base == "mr":
        return ", ".join(cleaned[:-1]) + " आणि " + cleaned[-1]
    if base == "hi" or language == "raj-IN":
        return ", ".join(cleaned[:-1]) + " और " + cleaned[-1]
    return ", ".join(cleaned[:-1]) + " and " + cleaned[-1]


def _join_sentences(pieces: list[str]) -> str:
    """Concatenate whole sentences with a space -- never with a conjunction.

    `_join` on two sentences produced "…teaching hospital. and Hostel residence
    is mandatory.", which no TTS voice can read naturally.
    """
    out = [p.strip() for p in pieces if p and p.strip()]
    if not out:
        return ""
    return " ".join(p if p.endswith((".", "!", "?", "।")) else p + "." for p in out)


def _tidy(text: str) -> str:
    """Clean up composed speech before it reaches TTS.

    Frames and knowledge-base values both end in punctuation, so naive
    concatenation yields artefacts like "…50% aggregate.." and stray spaces
    before sentence marks.
    """
    text = re.sub(r"\s+", " ", text or "").strip()
    text = re.sub(r"\s+([.,!?;:।])", r"\1", text)          # no space before marks
    text = re.sub(r"([.!?।])\1+", r"\1", text)             # ".." -> "."
    text = re.sub(r"\.।", "।", text)                        # ".।" -> "।"
    return text.strip()


#: Periods that are *not* sentence ends. "BBA LL.B. honours", "B.Sc. programmes",
#: "Ph.D. technology" and "Survey No. 499" all put a full stop followed by a space
#: in the middle of a sentence, and the naive split on `(?<=[.!?।])\s+` cut them in
#: half — the caller heard "Programmes such as B.Com honours, BBA LL.B." and the
#: chunker cut the same sentence across two chunks. A capitalised token of up to
#: three letters followed by a lowercase word or a number is an abbreviation, so its
#: period is protected while splitting and restored afterwards.
_ABBREV_PERIOD_RE = re.compile(r"(?<![A-Za-z])([A-Z][A-Za-z]{0,2})\.(?=\s+[a-z0-9])")
_ABBREV_SENTINEL = "\x00"
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?\u0964])\s+")


def split_sentences(text: str) -> list[str]:
    """Split on real sentence boundaries, keeping abbreviations intact."""
    protected = _ABBREV_PERIOD_RE.sub(lambda m: m.group(1) + _ABBREV_SENTINEL, text)
    return [
        part.replace(_ABBREV_SENTINEL, ".").strip()
        for part in _SENTENCE_SPLIT_RE.split(protected)
    ]


def _sentences(text: str, limit: int = 2, max_chars: int = 220) -> list[str]:
    text = re.sub(r"\s+", " ", text or "").strip()
    if text.startswith("\u2026"):
        # An overlap chunk carries the tail of the previous one, marked with a
        # leading ellipsis, so its first "sentence" is a fragment
        # ("\u2026on campus or online; offers are made by the recruiter"). The
        # complete sentence lives in the neighbouring chunk; drop the fragment
        # rather than read half a thought aloud.
        trimmed = split_sentences(text[1:])
        text = " ".join(trimmed[1:]).strip() if len(trimmed) > 1 else ""
    if not text:
        return []
    parts = [p for p in split_sentences(text) if p]
    out: list[str] = []
    for part in parts:
        if len(part) < 12:
            continue
        if _FIELD_DUMP_RE.match(part):
            # A facts chunk is rendered as "Degree: B.Tech / Seats: 60 / Fees —
            # total fee INR 5.10 lakh." Those lines exist so BM25 can match a
            # query; read aloud they sound like a spreadsheet, and on a Marathi
            # call they splice raw English into a Devanagari answer.
            continue
        out.append(part if part.endswith((".", "!", "?", "।")) else part + ".")
        if len(out) >= limit:
            break
    return [s[:max_chars] for s in out]


#: Sentence openers that are artefacts of how facts chunks are rendered for
#: retrieval ("Also known as: …", "Accreditation: NAAC"). Aliases and labelled
#: key/value dumps help BM25 match a query but must never be read aloud.
_LABEL_DUMP_RE = re.compile(
    r"^\s*(also known as|aliases?|tags?|slug|verified( by)?|source|as of|"
    r"effective (from|to)|last updated|academic year)\s*[:\-]",
    re.IGNORECASE,
)

#: Any "Label: value" line (max four label words) and the renderer's
#: "Fees — total fee …" form. This is the choke point for spoken text, so the
#: filter lives in `_sentences` rather than in each compose branch.
_FIELD_DUMP_RE = re.compile(
    r"^\s*(?:[A-Z][\w&/().\-]*(?:\s+[\w&/().\-]+){0,3}\s*:\s+\S"
    r"|Fees\s*[—–-]\s+\S)"
)


_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
_LATIN_RE = re.compile(r"[A-Za-z]")


def _script_compatible(sentence: str, language: str | None) -> bool:
    """True unless the sentence would splice a foreign script into the answer.

    The seeded KB is English, so on a Marathi or Hindi call the extractive
    fallback would otherwise tack an English paragraph onto the end of a native
    sentence frame ("होय, वसतिगृह उपलब्ध आहे… Confirm the current room types").
    Frame payloads (eligibility text, fee notes) are exempt — those *are* the
    fact being conveyed — this only filters prose lifted out of a chunk.
    """
    if not language:
        return True
    base = language.split("-")[0]
    if base in {"hi", "mr"} or language == "raj-IN":
        return bool(_DEVANAGARI_RE.search(sentence)) or not _LATIN_RE.search(sentence)
    # Symmetric guard: on an English call a Devanagari sentence is just as much a
    # splice. It used to slip through from the localised-title chunks the KB now
    # carries ("…School of Technology, Management & Engineering. बी.टेक संगणक
    # अभियांत्रिकी."), which read out as gibberish to an English caller.
    return not _DEVANAGARI_RE.search(sentence)


def _best_sentences(
    query: str,
    chunk: RetrievedChunk,
    limit: int = 2,
    language: str | None = None,
) -> list[str]:
    """Extractive fallback: pick the sentences that overlap the query most."""
    _header, _, body = chunk.text.partition("\n")
    # max_chars is a *filter* here, not a truncation: `_sentences` would otherwise
    # hand back a sentence cut mid-clause, and the composer's own 320-char budget
    # then cut that fragment again — the caller heard "Programmes such as B.Com
    # honours, BBA LL.B." because "LL.B." looks like a sentence end. A long
    # sentence is better sent by message than read aloud half-finished.
    candidates = [
        s for s in _sentences(body or chunk.text, limit=8, max_chars=10_000)
        if not _LABEL_DUMP_RE.match(s)
        and _script_compatible(s, language)
        and len(s) <= 240
    ]
    if not candidates:
        return []
    query_tokens = {t for t in re.findall(r"\w{2,}", query.lower())}
    scored = sorted(
        candidates,
        key=lambda s: -len(query_tokens & set(re.findall(r"\w{2,}", s.lower()))),
    )
    # keep original order for coherence
    picked = [s for s in candidates if s in scored[:limit]][:limit]
    return picked


def compose(
    retrieval: RetrievalResult,
    *,
    language: str,
    question: str,
    intent: IntentResult | None = None,
    min_confidence: float = 0.35,
) -> ComposedAnswer:
    """Build a spoken answer from retrieved records. Never invents a number."""
    intent = intent or retrieval.intent
    frames = _frames(language)
    fallback_language = language not in FRAMES and language.split("-")[0] not in {
        c.split("-")[0] for c in FRAMES
    }
    citations = retrieval.citations
    items = retrieval.items

    if not items or not retrieval.grounded:
        return ComposedAnswer(
            text=frames["not_found"],
            confidence=round(max(0.0, retrieval.best_score), 3),
            grounded=False,
            needs_escalation=True,
            escalation_reason="kb_no_answer",
            intent=intent.intent,
            language=language,
            fallback_language=fallback_language,
            citations=citations,
            template="not_found",
            debug={"best_score": retrieval.best_score},
        )

    groups = _group_by_record(items)
    primary, primary_chunks = groups[0]
    structured = _merge_structured(primary_chunks)
    # `programme` is spliced into frames as a *programme name*. When the record that
    # won is a policy or FAQ record, its title is a statement ("Fees are not
    # published on the university website"), and the fees frame came out as "Fees
    # are not published on the university website ची अधिकृत फी माझ्याकडे नोंदलेली
    # नाही." Only a course record's title is a programme name.
    is_programme_record = primary.category in PROGRAMME_CATEGORIES or bool(
        structured.get("programme") or structured.get("degree")
    )
    programme = (
        structured.get("programme")
        or structured.get("degree")
        or (primary.title if is_programme_record else "")
        or PROGRAMME_NEUTRAL.get(language.split("-")[0], "this programme")
    )
    verified = primary.verified
    stale = bool(getattr(primary, "signals", {}).get("stale"))
    sentences: list[str] = []
    followup: dict[str, Any] | None = None
    template_used = "generic"
    confidence = min(0.95, 0.45 + retrieval.best_score * 0.6)
    needs_escalation = False
    escalation_reason: str | None = None

    def _field_from_groups(key: str) -> tuple[Any, Any]:
        """Return `(value, chunk)` for `key`, searching the other retrieved records.

        A generic record can outrank the specific one: for "what is the eligibility
        for MCA", "Eligibility summary by level" beat the MCA record on category
        match, and the summary carries no `eligibility` field — so the call
        escalated even though the fact was sitting two ranks down in the same
        retrieval. The caller gets the fact, and the programme name comes from the
        record that actually holds it.
        """
        if structured.get(key):
            return structured[key], primary
        for chunk, chunks in groups[1:]:
            merged = _merge_structured(chunks)
            if merged.get(key):
                return merged[key], chunk
        return None, primary

    fees = structured.get("fees") if isinstance(structured.get("fees"), dict) else {}
    # No `per=` here: the fees_* frames already say "per year" / "प्रति वर्ष",
    # and passing it twice produced "one lakh forty thousand rupees per year per year".
    annual = _money(fees.get("annual") or fees.get("year_1"), language)
    total = _money(fees.get("total"), language)
    hostel_fee = _money(fees.get("hostel"), language)

    if structured.get("not_offered"):
        # "We do not run that" is a complete answer to any question about the
        # programme it denies — process, fees, eligibility or seats alike. Letting
        # the intent frame wrap it produced "एडमिशन की प्रक्रिया है: SVKM NMIMS
        # Global University does not run MBBS, BDS or any dental programme…" for a
        # Hindi caller who asked how to get admission to MBBS.
        sentences.extend(_best_sentences(question, primary, limit=3, language=language))
        template_used = "not_offered"
        alternatives = structured.get("offered_instead")
        if isinstance(alternatives, str):
            alternatives = [x.strip() for x in re.split(r"[;\n]", alternatives) if x.strip()]
        if alternatives:
            followup = {
                "channel": "sms",
                "title": "Programmes offered instead",
                "items": [str(a) for a in alternatives][:6],
            }

    elif intent.intent == "fees":
        if annual and total:
            sentences.append(frames["fees_both"].format(programme=programme, annual=annual, total=total))
            template_used = "fees_both"
        elif annual:
            sentences.append(frames["fees_annual"].format(programme=programme, annual=annual))
            template_used = "fees_annual"
        elif total:
            sentences.append(frames["fees_total"].format(programme=programme, total=total))
            template_used = "fees_total"
        elif structured.get("published_fee_table") is False or not is_programme_record:
            # The record that answered is the university's own statement that no fee
            # table is published. That is a different answer from "we have no figure
            # for programme X", and it needs no programme name in it.
            sentences.append(frames["fees_not_published"])
            template_used = "fees_not_published"
            needs_escalation = True
            escalation_reason = "fee_not_in_kb"
            where = structured.get("where_the_fee_is_confirmed")
            if isinstance(where, str):
                where = [x.strip() for x in re.split(r"[;\n]", where) if x.strip()]
            if where:
                followup = {
                    "channel": "sms",
                    "title": "Where the fee is confirmed",
                    "items": [str(w) for w in where][:6],
                }
        else:
            # No fee figure is recorded for this programme. Saying so in the
            # caller's own language and handing over beats reading English field
            # dumps — and beats inventing a number, which is the one thing this
            # assistant must never do on a live admissions line.
            sentences.append(frames["fees_not_recorded"].format(
                programme=_caller_named_only_the_degree(question, str(programme)) or programme))
            template_used = "fees_not_recorded"
            needs_escalation = True
            escalation_reason = "fee_not_in_kb"
        if hostel_fee and re.search(r"hostel|हॉस्टल", question, re.IGNORECASE):
            sentences.append(frames["fees_hostel"].format(hostel=hostel_fee))
        extra_fees = [k for k in fees if k not in {"annual", "year_1", "total", "hostel", "note", "deposit", "application", "exam"}]
        if extra_fees or fees.get("note"):
            followup = {
                "channel": "whatsapp",
                "title": f"{programme} — fee breakup",
                "items": [
                    f"{k.replace('_',' ')}: {_money(fees[k], 'en-IN')}" for k in extra_fees
                ] + ([str(fees["note"])] if fees.get("note") else []),
            }
        if not verified and template_used in {"fees_both", "fees_annual", "fees_total"}:
            # Only meaningful after a figure has actually been spoken; on the
            # fees_not_recorded path it contradicted the sentence before it.
            sentences.append(frames["fees_unverified"])
            confidence *= 0.8

    elif intent.intent == "loan_payment":
        # Loans, refunds and withdrawals are all things the university publishes
        # nothing about. The record says so in English prose only, which a Marathi
        # or Hindi caller never heard: the call escalated with a bare "I don't
        # have that information". Say the true thing in the caller's language.
        not_published = any(
            structured.get(flag) is False
            for flag in ("published", "published_by_university", "published_fee_table")
        )
        subject = (primary.title or "").lower()
        if not_published and ("refund" in subject or "withdraw" in subject):
            sentences.append(frames["refund_not_published"])
            template_used = "refund_not_published"
        elif not_published:
            sentences.append(frames["loan_not_published"])
            template_used = "loan_not_published"
        else:
            sentences.extend(_best_sentences(question, primary, language=language))
            template_used = "generic"
        if not_published:
            needs_escalation = True
            escalation_reason = "not_published"
            office = structured.get("who_handles_this") or structured.get("contact")
            items = [str(structured[k]) for k in ("what_the_university_can_issue", "important")
                     if isinstance(structured.get(k), str)]
            if isinstance(office, str):
                items.append(office)
            if items:
                followup = {
                    "channel": "sms",
                    "title": primary.title or "Accounts and loans",
                    "items": items[:6],
                }

    elif intent.intent == "eligibility":
        eligibility, source = _field_from_groups("eligibility")
        if eligibility and not _programme_named(intent):
            # "What is the eligibility?" names no programme, but retrieval still
            # ranks one first and the frame speaks its title, so a caller who
            # never said Mechanical was told "For B.Tech Mechanical Engineering,
            # the eligibility is…". Eligibility is what a caller decides whether
            # to apply on, so the wrong programme does real damage; asking costs
            # one turn.
            sentences.append(frames["ask_clarify"].format(
                programme=PROGRAMME_NOUN.get(language.split("-")[0], "programme")
            ))
            template_used = "ask_clarify"
        elif eligibility:
            if source is not primary and source.category in PROGRAMME_CATEGORIES:
                programme = source.title
            payload = str(eligibility).strip()
            # "…with at least 50% marks in aggregate. Mathematics is not
            # compulsory." already ends its own sentence, so the short frame's
            # trailing "चाहिए"/"आवश्यक आहे" landed after the full stop and read
            # out as a dangling fragment. Long or self-terminated payloads take
            # the colon frame instead, which has no trailing verb.
            key = (
                "eligibility_long"
                if len(payload) > 90 or payload.endswith((".", "!", "?", "\u0964"))
                else "eligibility"
            )
            sentences.append(frames[key].format(programme=programme, eligibility=payload))
            template_used = key
            exam = structured.get("entrance_exam")
            if exam:
                sentences.append(frames["eligibility_exam"].format(exam=exam))
        else:
            sentences.extend(_best_sentences(question, primary, language=language))
            template_used = "generic"

    elif intent.intent == "entrance_exam":
        # The tests record keeps one entry per programme under `tests`; a course
        # record keeps its own under `entrance_exam`. Read the record that won
        # first -- reaching into another record produced "round-wise schedules and
        # merit lists are published..." as the answer to "which exam do I give?".
        tests = structured.get("tests")
        if isinstance(tests, str):
            tests = [x.strip() for x in re.split(r"[;\n]", tests) if x.strip()]
        if tests:
            wanted = [t.lower().replace(".", "") for t in (intent.course_tokens or [])]
            match = next(
                (
                    entry for entry in tests
                    if any(w and w in str(entry).lower().replace(".", "") for w in wanted)
                ),
                None,
            )
            if match:
                sentences.append(str(match))
                template_used = "exam"
            else:
                sentences.append(
                    frames["entrance_tests_generic"].format(examples=str(tests[0]))
                )
                template_used = "entrance_tests_generic"
                followup = {
                    "channel": "sms",
                    "title": "Entrance tests by programme",
                    "items": [str(t) for t in tests],
                }
            exam, source = None, primary
            tests_answered = True
        else:
            exam, source = _field_from_groups("entrance_exam")
            if not exam:
                exam, source = _field_from_groups("selection_process")
            tests_answered = False
        if exam:
            if source is not primary and source.category in PROGRAMME_CATEGORIES:
                programme = source.title
            sentences.append(str(exam))
            sentences.extend(_sentences(primary.text.partition("\n")[2], limit=1))
            template_used = "exam"
        elif not tests_answered:
            sentences.extend(_best_sentences(question, primary, language=language))
            template_used = "generic"

    elif intent.intent == "admission_process":
        steps = structured.get("selection_process") or structured.get("steps")
        if isinstance(steps, str):
            parts = [p.strip(" .") for p in re.split(r"[;|\n]|\.\s", steps) if p.strip()]
        elif isinstance(steps, list):
            parts = [str(p) for p in steps]
        else:
            parts = _sentences(primary.text.partition("\n")[2], limit=3)
        if parts:
            short = parts[:3]
            # `_join` is for short labels ("B.Tech, BBA आणि MBA"). When the KB
            # gives prose instead, joining it with a conjunction splices Marathi
            # into an English sentence mid-clause, so keep the sentences apart.
            joiner = _join_sentences if any(len(p) > 60 for p in short) else _join
            steps_text = (
                joiner(short) if joiner is _join_sentences else _join(short, language)
            )
            sentences.append(frames["admission_process"].format(steps=steps_text))
            template_used = "admission_process"
            if len(parts) > 3:
                followup = {
                    "channel": "whatsapp", "title": "Admission process — full steps",
                    "items": parts,
                }
        else:
            sentences.extend(_best_sentences(question, primary, language=language))
            template_used = "generic"

    elif intent.intent == "important_dates":
        dates = structured.get("important_dates")
        lines: list[str] = []
        if isinstance(dates, dict):
            lines = [f"{k.replace('_',' ')}: {v}" for k, v in list(dates.items())[:3]]
        elif isinstance(dates, list):
            lines = [str(d) for d in dates[:3]]
        elif isinstance(dates, str):
            lines = [s.strip() for s in re.split(r"[;\n]", dates) if s.strip()][:3]
        mechanism = structured.get("how_it_works") or structured.get("published_for_ay_2026_27")
        rounds = structured.get("rounds_running_for_ay_2026_27") or structured.get("rounds_published_for_ay_2026_27")
        round_entries = (
            [str(r) for r in rounds] if isinstance(rounds, list)
            else ([str(rounds)] if rounds else [])
        )
        if not lines and (mechanism or rounds):
            # AY 2026-27 dates are published per programme and per round as PDFs on
            # the website, so no single date can be spoken. When the caller named a
            # programme, read back that programme's published round status -- "which
            # round is MCA admission in?" deserves "up to round two", not the
            # generic mechanism sentence. Otherwise explain the mechanism in the
            # caller's language and send the schedules by SMS.
            wanted = [t.lower().replace(".", "") for t in (intent.course_tokens or [])]
            named = next(
                (
                    entry for entry in round_entries
                    if ":" in entry
                    and any(w and w in entry.lower().replace(".", "") for w in wanted)
                ),
                None,
            ) if wanted else None
            if named:
                name, _, rest = named.partition(":")
                sentences.append(frames["round_status"].format(
                    programme=name.strip(), status=rest.strip() or named))
                template_used = "round_status"
            else:
                sentences.append(frames["important_dates_mechanism"])
                template_used = "important_dates_mechanism"
            items: list[str] = []
            if mechanism:
                items.append(str(mechanism))
            items.extend(round_entries)
            if structured.get("how_to_get_current_dates"):
                items.append(str(structured["how_to_get_current_dates"]))
            if items:
                followup = {
                    "channel": "sms",
                    "title": "Admission schedules — AY 2026-27",
                    "items": items[:6],
                }
        elif lines:
            # Date lines written as full sentences must not be glued together with
            # ", " and " and ": the calendar record produced one 320-character
            # run-on that the trim then cut in the middle of a date. Short labels
            # keep the conjunction join; long prose lines are spoken as separate
            # sentences, two at a time, with the rest sent by SMS.
            spoken_lines = lines if max((len(x) for x in lines), default=0) <= 60 else lines[:2]
            if spoken_lines is lines:
                dates_text = _join(spoken_lines, language)
            else:
                dates_text = " ".join(
                    ln.strip().rstrip(".") + "." for ln in spoken_lines
                )
            sentences.append(frames["important_dates"].format(dates=dates_text))
            template_used = "important_dates"
            if isinstance(dates, (dict, list)) and len(dates) > 3:
                followup = {
                    "channel": "sms", "title": "NMIMS Dhule important dates",
                    "items": [str(v) for v in (dates.values() if isinstance(dates, dict) else dates)],
                }
        else:
            needs_escalation = True
            escalation_reason = "kb_no_answer"
            sentences = [frames["not_found"]]

    elif intent.intent == "scholarships":
        types = structured.get("scholarship_types") or []
        if isinstance(types, str):
            types = [t.strip() for t in re.split(r"[;,\n]", types) if t.strip()]
        if types:
            detail = [
                s for s in _sentences(primary.text.partition("\n")[2], limit=2)
                if _script_compatible(s, language)
            ]
            # `detail` is prose lifted from the facts chunk; on a non-Latin call
            # the script filter removes it, so don't leave a trailing gap.
            text = frames["scholarships"].format(
                types=_join(list(types)[:3], language),
                detail=detail[0] if detail else "",
            ).strip()
            sentences.append(re.sub(r"\s{2,}", " ", text))
            template_used = "scholarships"
        else:
            sentences.extend(_best_sentences(question, primary, language=language))
            template_used = "generic"

    elif intent.intent == "hostel":
        # The university website publishes no hostel information at all, so the KB
        # records that explicitly rather than borrowing a claim from an aggregator
        # listing for another institute on the same campus.
        status = str(structured.get("hostel_status") or "").lower()
        if status == "not_published" or structured.get("published_on_website") is False:
            sentences.append(frames["hostel_unconfirmed"])
            template_used = "hostel_unconfirmed"
            needs_escalation = True
            escalation_reason = "hostel_not_published"
            phones = structured.get("school_phones")
            if isinstance(phones, str):
                phones = [x.strip() for x in re.split(r"[;\n]", phones) if x.strip()]
            items: list[str] = [str(x) for x in (phones or [])]
            if structured.get("who_confirms"):
                items.append(str(structured["who_confirms"]))
            if items:
                followup = {
                    "channel": "sms",
                    "title": "Hostel — confirm with the school office",
                    "items": items[:6],
                }

        else:
            summary = structured.get("summary") or structured.get("description")
            rooms = structured.get("room_types")
            if isinstance(rooms, str):
                rooms = [r.strip() for r in re.split(r"[;,\n]", rooms) if r.strip()]
            available = structured.get("hostel_available")
            pieces: list[str] = []
            if isinstance(available, bool):
                # The KB records availability as a fact, so it can be stated in the
                # caller's own language instead of reading English prose back to a
                # Marathi caller. Charges are not in the KB, so they are offered as a
                # follow-up rather than spoken.
                pieces.append(frames["hostel_available" if available else "hostel_none"])
                template_used = "hostel_available" if available else "hostel_none"
                charges = structured.get("charges")
                if isinstance(charges, dict) and charges.get("note"):
                    followup = {
                        "channel": "sms",
                        "title": f"{programme} — hostel charges",
                        "items": [str(charges["note"])],
                    }
            elif summary:
                pieces.extend(_sentences(str(summary), limit=2))
            if rooms:
                spoken_rooms = [r for r in list(rooms)[:3] if _script_compatible(r, language)]
                if spoken_rooms:
                    pieces.append(_join(spoken_rooms, language))
            if hostel_fee:
                pieces.append(frames["fees_hostel"].format(hostel=hostel_fee))
            if not pieces:
                pieces = _best_sentences(question, primary, language=language)
            if template_used != "hostel":
                sentences.extend(pieces[:2])
            else:
                sentences.append(frames["hostel"].format(summary=_join_sentences(pieces[:2])))
            template_used = template_used if template_used in {
                "hostel_available", "hostel_none"
            } else "hostel"

    elif intent.intent == "placements":
        highest = _money(structured.get("highest_package"), language)
        average = _money(structured.get("average_package") or structured.get("median_package"), language)
        recruiters = structured.get("recruiters") or []
        if isinstance(recruiters, str):
            recruiters = [r.strip() for r in re.split(r"[;,\n]", recruiters) if r.strip()]
        pieces = []
        if highest or average:
            pieces.append(
                frames["placement_numbers"].format(
                    highest=highest or "-", average=average or "-"
                )
            )
        if recruiters:
            pieces.append(frames["recruiters"].format(names=_join(list(recruiters)[:3], language)))
            if len(recruiters) > 3:
                followup = {
                    "channel": "sms", "title": f"{programme} recruiters",
                    "items": [str(r) for r in recruiters],
                }
        if not pieces and _PLACEMENT_DETAIL_RE.search(question or "") and (
            structured.get("published_detail") is False or structured.get("not_published")
        ):
            # "What is the highest package?" was answered with the homepage's
            # "100% job placement" claim, which says nothing about a package. The
            # record lists exactly what is not published, so say that and hand over
            # rather than letting an unrelated claim stand in for the answer.
            sentences.append(frames["placements_not_published"])
            template_used = "placements_not_published"
            needs_escalation = True
            escalation_reason = "not_published"
            missing = structured.get("not_published")
            if isinstance(missing, str):
                missing = [x.strip() for x in re.split(r"[;\n]", missing) if x.strip()]
            if missing:
                followup = {
                    "channel": "sms", "title": "Placements — not published",
                    "items": [str(m) for m in missing][:6],
                }
        else:
            if not pieces:
                pieces = _best_sentences(question, primary, language=language)
                sentences.append(frames["placements"].format(summary=" ".join(pieces[:2])))
            template_used = "placements"

    elif intent.intent == "contact":
        phone = structured.get("contact_phone")
        email = structured.get("contact_email")
        base = language.split("-")[0]
        localised = structured.get("department_localized")
        department = (
            (localised.get(language) if isinstance(localised, dict) else None)
            or (localised.get(base) if isinstance(localised, dict) else None)
            or structured.get("department")
            or structured.get("school")
            # Not `programme`: that is the record title, so the frame came out as
            # "You can reach Admissions contact details and helpline on …".
            # `language` is a full locale code (mr-IN), so key on the base.
            or DEPARTMENT_FALLBACK.get(base, "the admissions office")
        )
        address = structured.get("address")
        if address and _ADDRESS_RE.search(question or ""):
            # "धुले कैंपस का पता बताओ" was answered with the landmarks and a pin
            # code spelled out in words, because the address record carries no
            # phone number and the branch fell through to extractive prose.
            sentences.append(frames["campus_address"].format(address=str(address)))
            template_used = "campus_address"
            card_items = [str(address)]
            for label, key in (("Landmark", "landmark"), ("Pin code", "pin_code"),
                               ("Website", "website")):
                if structured.get(key):
                    card_items.append(f"{label}: {structured[key]}")
            followup = {
                "channel": "sms",
                "title": "SVKM NMIMS Global University, Dhule — address",
                "items": card_items[:6],
            }
        elif phone and email:
            sentences.append(
                frames["contact"].format(department=department, phone=phone, email=email)
            )
            template_used = "contact"
        elif phone:
            # The university publishes phone numbers but no email, so the department
            # is named in the caller's own language next to the number instead of
            # the frame being reduced to "The number is …".
            sentences.append(
                frames["contact_department_phone"].format(department=department, phone=phone)
            )
            template_used = "contact_phone"
        elif email:
            sentences.append(frames["contact_email"].format(email=email))
            template_used = "contact_email"
        if template_used == "campus_address":
            # The address frame is the whole answer and its SMS card is already
            # set. Without this the branch below appended extractive prose and
            # overwrote the template name with "generic", so a caller asking for
            # the address heard the address and then the landmarks again.
            pass
        elif template_used in {"contact", "contact_phone", "contact_email"} and not followup:
            # One number is easy to say aloud; the full card is easier to read.
            # The website publishes school-wise numbers only (no email, no toll
            # free line), so the card lists every school number plus the portal and
            # the website rather than one number read aloud.
            card: list[str] = []
            school_phones = structured.get("school_phones")
            if isinstance(school_phones, str):
                school_phones = [x.strip() for x in re.split(r"[;\n]", school_phones) if x.strip()]
            for line in school_phones or []:
                card.append(str(line))
            for label, key in (
                ("Enquiry form", "enquiry_form"),
                ("Admission portal", "admission_portal"),
                ("Website", "website"),
            ):
                value = structured.get(key)
                if value:
                    card.append(f"{label}: {value}")
            if card:
                followup = {
                    "channel": "sms",
                    "title": "SVKM NMIMS Global University, Dhule — contact details",
                    "items": card,
                }
        else:
            sentences.extend(_best_sentences(question, primary, language=language))
            template_used = "generic"

    elif intent.intent == "documents":
        docs = structured.get("documents_required") or []
        if isinstance(docs, str):
            docs = [d.strip() for d in re.split(r"[;,\n]", docs) if d.strip()]
        if docs:
            sentences.append(frames["documents"].format(items=_join(list(docs)[:3], language)))
            template_used = "documents"
            if len(docs) > 3:
                followup = {
                    "channel": "whatsapp", "title": "Documents required",
                    "items": [str(d) for d in docs],
                }
        else:
            sentences.extend(_best_sentences(question, primary, language=language))
            template_used = "generic"

    elif intent.intent in {"facilities", "transport"}:
        transport_mode = _transport_mode(question) if intent.intent == "transport" else None
        if intent.intent == "transport" and transport_mode == "rail" and structured.get("by_rail"):
            sentences.append(frames["transport_rail"].format(detail=str(structured["by_rail"])))
            template_used = "transport_rail"
            rail_items = [str(structured["by_rail"])]
            if structured.get("on_highway"):
                rail_items.append(str(structured["on_highway"]))
            if structured.get("nearest_airport"):
                rail_items.append(f"Nearest airport: {structured['nearest_airport']}")
            followup = {
                "channel": "sms", "title": "Reaching the Dhule campus",
                "items": rail_items[:6],
            }
        elif intent.intent == "transport" and transport_mode == "air" and structured.get("by_air"):
            sentences.append(frames["transport_air"].format(detail=str(structured["by_air"])))
            template_used = "transport_air"
            air_items = [str(structured["by_air"])]
            if structured.get("nearest_airport"):
                air_items.append(f"Nearest airport: {structured['nearest_airport']}")
            followup = {
                "channel": "sms", "title": "Reaching the Dhule campus",
                "items": air_items[:6],
            }
        elif intent.intent == "transport" and transport_mode == "road" and structured.get("road_distances"):
            road = structured["road_distances"]
            if isinstance(road, str):
                road = [x.strip() for x in re.split(r"[;\n]", road) if x.strip()]
            sentences.append(frames["transport_road"].format(
                distances=_join([str(d) for d in road[:3]], language)))
            if structured.get("on_highway"):
                sentences.append(str(structured["on_highway"]))
            template_used = "transport_road"
            road_items = [str(d) for d in road]
            if structured.get("on_highway"):
                road_items.append(str(structured["on_highway"]))
            if structured.get("nearest_airport"):
                road_items.append(f"Nearest airport: {structured['nearest_airport']}")
            followup = {
                "channel": "sms", "title": "Reaching the Dhule campus",
                "items": road_items[:6],
            }
        elif intent.intent == "transport" and structured.get("nearest_airport") and structured.get("road_distances"):
            # Verified travel data from the university's own "How to reach" section:
            # name the airport and a few road distances, then send the rest by SMS.
            distances = structured["road_distances"]
            if isinstance(distances, str):
                distances = [x.strip() for x in re.split(r"[;\n]", distances) if x.strip()]
            airport = str(structured["nearest_airport"]).split(",")[0].strip()
            sentences.append(frames["transport_summary"].format(
                airport=airport,
                distances=_join([str(d) for d in distances[:3]], language),
            ))
            template_used = "transport_summary"
            items = [str(d) for d in distances]
            items.append(f"Nearest airport: {structured['nearest_airport']}")
            if structured.get("by_rail"):
                items.append(f"By rail: {structured['by_rail']}")
            if structured.get("on_highway"):
                items.append(str(structured["on_highway"]))
            followup = {
                "channel": "sms",
                "title": "Reaching the Dhule campus",
                "items": items[:6],
            }

        else:
            fac = structured.get("facilities") or []
            if isinstance(fac, str):
                fac = [f.strip() for f in re.split(r"[;,\n]", fac) if f.strip()]
            frame_key = "facilities" if intent.intent == "facilities" else "transport"
            if fac and frame_key == "facilities":
                sentences.append(frames["facilities"].format(items=_join(list(fac)[:3], language)))
                template_used = "facilities"
                if len(fac) > 3:
                    followup = {"channel": "sms", "title": "Campus facilities", "items": [str(f) for f in fac]}
            else:
                body = _best_sentences(question, primary, language=language)
                if body:
                    frame_text = frames[frame_key]
                    if "{summary}" in frame_text:
                        # The transport frame is a bare "{summary}", so prose can go
                        # straight in.
                        sentences.append(frame_text.format(summary=" ".join(body)))
                    else:
                        # The facilities frame is "The campus has {items}." Wrapping a
                        # whole prose sentence in it produced "The campus has The
                        # campus is directly on the Mumbai Agra Highway…", so speak
                        # the prose on its own instead.
                        sentences.extend(body[:2])
                    template_used = frame_key
                # If the script filter emptied `body` (an English-only KB answering a
                # Marathi caller), say nothing here: the fallback pass at the end of
                # compose answers in English and flags the degradation, which beats
                # reading a bare record title aloud.

    elif intent.intent == "comparison" and len(groups) >= 2:
        left, left_chunks = groups[0]
        right, right_chunks = groups[1]
        left_fees = _merge_structured(left_chunks).get("fees", {})
        right_fees = _merge_structured(right_chunks).get("fees", {})
        left_fee = _money(left_fees.get("total") or left_fees.get("annual"), language)
        right_fee = _money(right_fees.get("total") or right_fees.get("annual"), language)
        if left_fee and right_fee:
            sentences.append(
                frames["comparison"].format(
                    left=left.title, left_fee=left_fee, right=right.title, right_fee=right_fee
                )
            )
            template_used = "comparison"
        else:
            sentences.extend(_best_sentences(question, left, language=language))
            template_used = "generic"

    elif intent.intent == "courses" and not intent.course_tokens:
        # Only a programme record's title is a course name. With a policy record in
        # the retrieved set the catalogue used to say "We offer Programme transfer
        # and deferral are not published, and many more programmes."
        def _is_programme_group(group: tuple[Any, Any]) -> bool:
            head = group[0]
            payload = getattr(head, "structured", None) or {}
            return head.category in PROGRAMME_CATEGORIES or bool(
                payload.get("programme") or payload.get("degree")
            )

        def _overview_programmes(group: tuple[Any, Any]) -> list[str] | None:
            """None when this is not an overview record at all.

            The caller has to tell "not an overview" from "an overview whose
            chunk carries no programme list", because only the second may be
            dropped. When it could not tell them apart it fell back to the
            title, and a caller asking how many seats there are heard "School of
            Pharmacy & Technology Management — programmes overview" offered as
            something they could enrol in.
            """
            """The real course names inside a school-overview record.

            An overview record's *title* is a school label ("School of Commerce —
            programmes overview"), not a course name, and the catalogue used to
            speak it as one: "We offer B.Tech (Cosmetic Technology), School of
            Commerce — programmes overview and Master of Pharmacy…". Its
            `programmes` list holds the names a caller can actually enrol in.
            """
            head = group[0]
            payload = getattr(head, "structured", None) or {}
            title = (head.title or "").lower()
            is_overview = (
                "overview" in title
                or "at a glance" in title
                or (
                    isinstance(payload.get("programmes"), (list, str))
                    and bool(payload.get("school"))
                    and not (payload.get("programme") or payload.get("degree"))
                )
            )
            if not is_overview:
                return None
            listed = payload.get("programmes") or []
            if isinstance(listed, str):
                listed = [x.strip() for x in re.split(r"[;\n]", listed) if x.strip()]
            names: list[str] = []
            for entry in listed:
                # Entries carry their own detail after a dash ("B.Pharm — 4 years,
                # semester, intake 60"); the catalogue speaks the name only.
                name = re.split(r"\s+\u2014\s+|\s+-\s+", str(entry))[0].strip()
                if name and name not in names:
                    names.append(name)
            return names[:4]

        def _spoken_name(title: str) -> str:
            """The part of a programme title a caller can say out loud.

            "Master of Pharmacy (M.Pharm) — Pharmaceutics, Quality Assurance,
            Pharmacology, Pharmaceutical Chemistry" is one record, but as a
            catalogue item it is a twenty-second mouthful. The detail after the
            dash goes in the follow-up message, which is where the caller can
            read it.
            """
            return re.split(r"\s+\u2014\s+|\s+-\s+", title or "")[0].strip()

        programme_groups = [g for g in groups if _is_programme_group(g)]
        titles: list[str] = []
        for group in programme_groups:
            listed = _overview_programmes(group)
            if listed is None:
                contributed = [_spoken_name(group[0].title)]
            else:
                # An overview with no names to give contributes nothing rather
                # than its own school label.
                contributed = [_spoken_name(name) for name in listed]
            for name in contributed:
                if name and name not in titles:
                    titles.append(name)
        if titles:
            sentences.append(frames["catalog"].format(items=_join(titles[:3], language)))
            template_used = "catalog"
            followup = {
                "channel": "whatsapp", "title": "NMIMS Global University, Dhule programmes",
                "items": titles[:12],
            }
        else:
            sentences.extend(_best_sentences(question, primary, language=language))
            template_used = "generic"

    else:
        # course overview: duration + seats + school, then extractive sentences
        duration = duration_phrase(structured.get("duration_years") or structured.get("duration"), language)
        seats = structured.get("seats")
        if duration and seats:
            sentences.append(
                frames["duration"].format(programme=programme, duration=duration,
                                          seats=spoken_count(seats, "", language))
            )
            template_used = "duration"
        elif duration:
            sentences.append(frames["duration_only"].format(programme=programme, duration=duration))
            template_used = "duration_only"
        skip_extract = False
        school = structured.get("school") or structured.get("department")
        if school and is_programme_record:
            sentences.append(frames["school"].format(school=school))
        # A named specialisation has its own intake. "What is the intake for
        # M.Pharm Pharmaceutics?" must not be answered with the 48 seats that
        # cover all four specialisations.
        specs = structured.get("specialisations")
        if isinstance(specs, str):
            specs = [x.strip() for x in re.split(r"[;\n]", specs) if x.strip()]
        if specs and intent.specialisations:
            asked = [a.lower() for a in intent.specialisations]
            matched_spec = next(
                (
                    entry for entry in specs
                    if any(a and a in str(entry).lower() for a in asked)
                ),
                None,
            )
            if matched_spec:
                # "Pharmaceutics, intake 15" is a label, and the spoken-length trim
                # re-splits the answer through a sentence filter that drops label
                # lines -- so the one number the caller asked for ended up in the
                # SMS instead of on the call. Speak it as a sentence, keep the
                # all-specialisation seat count out of it, and stop there.
                spec_text = str(matched_spec)
                name, _, rest = spec_text.partition("\u2014")
                number = re.search(r"(\d+)", rest or spec_text)
                sentences = [x for x in sentences if "seats" not in x.lower()]
                if duration:
                    sentences.insert(0, frames["duration_only"].format(
                        programme=programme, duration=duration))
                if number:
                    sentences.append(frames["specialisation_intake"].format(
                        spec=name.strip() or spec_text,
                        seats=spoken_count(number.group(1), "", language),
                    ))
                else:
                    sentences.append(name.strip() or spec_text)
                template_used = "specialisation_intake"
                followup = {
                    "channel": "sms",
                    "title": (programme or "Specialisations") + " \u2014 intake by specialisation",
                    "items": [str(x) for x in specs],
                }
                skip_extract = True
        # Every seed record carries a hand-written `answer`: one or two short
        # spoken sentences. Prefer it over slicing prose out of a facts chunk.
        spoken = None if skip_extract else structured.get("answer")
        extracted: list[str] = []
        if isinstance(spoken, str) and spoken.strip():
            extracted = _sentences(spoken.strip(), limit=2, max_chars=200)
            if extracted:
                template_used = "spoken_answer"
        if not extracted and not skip_extract:
            extracted = _best_sentences(question, primary, limit=2 if not sentences else 1,
                                    language=language)
        sentences.extend(extracted)
        # Deliberately no `sentences.append(programme)` here. `programme` is the
        # record title, and a title is a label, not an answer: a Hindi caller who
        # asked "क्या यहाँ एमबीबीएस है?" used to hear "Medical, dental, nursing and
        # allied health programmes are not offered" read out as the whole reply.
        # Leaving `sentences` empty falls through to the English-prose fallback
        # below, which says something true and flags the degradation for QA.
        if not extracted and not duration and not skip_extract:
            template_used = "generic"

    # eligibility is almost always useful right after fees -- but the programme
    # name was just spoken, so use the short frame instead of repeating it
    if (
        intent.intent == "fees"
        and structured.get("eligibility")
        and escalation_reason != "fee_not_in_kb"
        and len(" ".join(sentences)) < 240
    ):
        frame = frames.get("eligibility_follow") or frames["eligibility"]
        sentences.append(frame.format(programme=programme, eligibility=structured["eligibility"]))

    if not needs_escalation and any(
        structured.get(flag) is False
        for flag in ("published", "published_by_university", "published_fee_table",
                     "published_on_website")
    ):
        # The record's own claim is that the university publishes nothing here. A
        # refund, a loan, a hostel room or a fee cannot be settled from an absence
        # of data, so the caller gets a person even when the sentence was useful.
        needs_escalation = True
        escalation_reason = "not_published"

    if stale and not needs_escalation:
        sentences.append(frames["stale_warning"])
        confidence *= 0.85

    text = _tidy(" ".join(s for s in sentences if s))
    if len(text) > MAX_SPOKEN_CHARS:
        # Voice answers must stay short. Keep whole sentences only, and offer the
        # remainder as a WhatsApp/SMS follow-up rather than talking over the caller.
        kept: list[str] = []
        for sentence in _sentences(text, limit=6, max_chars=len(text)):
            if sum(len(k) for k in kept) + len(sentence) > MAX_SPOKEN_CHARS:
                break
            kept.append(sentence)
        if kept:
            text = _tidy(" ".join(kept))
        else:
            # A single sentence longer than the whole spoken budget. Cut at the
            # last clause boundary rather than mid-word: the caller hears an
            # unfinished but intelligible clause, and the follow-up message
            # already carries the full text.
            head = text[:MAX_SPOKEN_CHARS]
            cut = max(head.rfind(","), head.rfind(";"), head.rfind(" "))
            # Rstrip the cut point: leaving the comma produced "…up to round two
            # for M.Tech, MCA and direct second year B.Tech,." on the call.
            text = _tidy(head[:cut].rstrip(" ,;:") if cut > MAX_SPOKEN_CHARS // 2 else head)
        remainder = [_tidy(s) for s in sentences if s and s not in kept]
        if followup is None:
            followup = {
                "channel": "whatsapp",
                "title": (programme or "NMIMS Global University, Dhule") + " — details",
                "items": remainder[:6],
            }
        else:
            # De-duplicate: the fee branch already seeds `items` with the same
            # sentences, and a repeated line in an SMS looks like a bug.
            existing = [_tidy(str(i)) for i in (followup.get("items") or [])]
            extra = [i for i in remainder if i not in existing]
            followup["items"] = existing + extra[:6]
    if not text:
        # Script filtering can empty the answer when the KB only holds English
        # prose. Rather than escalating on a question we *can* answer, say it in
        # English and flag the degradation for QA.
        english = _best_sentences(question, primary, limit=2)
        if not english:
            # The record that won may be a facts dump (every line a label) while
            # its own prose sits in another chunk. Stay inside that record, or
            # inside the categories this intent asked for: reaching into whatever
            # ranked next made a Marathi caller asking about education loans hear
            # a description of the engineering school instead.
            same_record = [
                c for c in items
                if c is not primary
                and (getattr(c, "record_id", None) or c.title)
                == (getattr(primary, "record_id", None) or primary.title)
            ]
            same_topic = [
                c for c in items
                if c is not primary and c not in same_record
                and c.category in set(intent.categories or ())
            ]
            for chunk in same_record + same_topic:
                english = _best_sentences(question, chunk, limit=2)
                if english:
                    break
        if english:
            text = _tidy(" ".join(english))
            fallback_language = True
    if not text:
        text = frames["not_found"]
        needs_escalation = True
        escalation_reason = "kb_no_answer"
        confidence = 0.1

    if not verified and not needs_escalation and confidence < min_confidence + 0.15:
        needs_escalation = True
        escalation_reason = "low_confidence"

    if template_used == "ask_clarify":
        # A question back to the caller is the opposite of giving up on them.
        # The record that ranked first may well be an unpublishable one, which
        # would otherwise have the call transferred while the assistant is still
        # asking which programme they meant.
        needs_escalation = False
        escalation_reason = None

    return ComposedAnswer(
        text=text,
        confidence=round(min(0.95, max(0.05, confidence)), 3),
        grounded=True,
        needs_escalation=needs_escalation,
        escalation_reason=escalation_reason,
        followup=followup,
        intent=intent.intent,
        language=language,
        fallback_language=fallback_language,
        citations=citations,
        template=template_used,
        debug={
            "record_id": primary.record_id,
            "record_title": primary.title,
            "best_score": retrieval.best_score,
            "verified": verified,
            "structured_keys": sorted(structured.keys()),
        },
    )
