"""Azure Speech Services provider implementation."""

import asyncio
import base64
from typing import AsyncGenerator, Optional, List
import azure.cognitiveservices.speech as speechsdk

from packages.providers.interfaces import (
    SpeechProvider, SpeechRecognitionResult, LanguageIdentificationResult,
    TTSAudioChunk, SpeechEvent
)
from packages.schemas.schemas import LanguageCode
from packages.config.settings import get_settings


class AzureSpeechProvider(SpeechProvider):
    """Azure Speech Services implementation."""

    def __init__(self):
        self.settings = get_settings()
        self._speech_config = speechsdk.SpeechConfig(
            subscription=self.settings.azure_speech.key,
            region=self.settings.azure_speech.region
        )
        self._active_recognizers = {}
        self._active_synthesizers = {}

    def _get_voice_name(self, language: LanguageCode) -> str:
        """Get Azure voice name for language."""
        voice_map = {
            LanguageCode.EN_IN: self.settings.conversation.tts_voice_en_in,
            LanguageCode.HI_IN: self.settings.conversation.tts_voice_hi_in,
            LanguageCode.MR_IN: self.settings.conversation.tts_voice_mr_in,
        }
        return voice_map.get(language, voice_map[LanguageCode.EN_IN])

    async def recognize_continuous(
        self,
        audio_stream: AsyncGenerator[bytes, None],
        language: Optional[LanguageCode] = None
    ) -> AsyncGenerator[SpeechRecognitionResult, None]:
        """Continuous speech recognition from audio stream."""
        # Create push stream
        stream = speechsdk.audio.PushAudioInputStream()
        audio_config = speechsdk.audio.AudioConfig(stream=stream)

        # Configure recognizer
        if language:
            self._speech_config.speech_recognition_language = language.value

        recognizer = speechsdk.SpeechRecognizer(
            speech_config=self._speech_config,
            audio_config=audio_config
        )

        # Result queue
        result_queue = asyncio.Queue()

        def recognizing_cb(evt):
            result_queue.put_nowait(SpeechRecognitionResult(
                text=evt.result.text,
                is_final=False,
                confidence=None,
                language=language,
                event_type=SpeechEvent.RECOGNIZING
            ))

        def recognized_cb(evt):
            if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                result_queue.put_nowait(SpeechRecognitionResult(
                    text=evt.result.text,
                    is_final=True,
                    confidence=None,  # Azure doesn't provide confidence in continuous mode
                    language=language,
                    event_type=SpeechEvent.RECOGNIZED
                ))

        def canceled_cb(evt):
            result_queue.put_nowait(None)  # Signal end

        recognizer.recognizing.connect(recognizing_cb)
        recognizer.recognized.connect(recognized_cb)
        recognizer.canceled.connect(canceled_cb)
        recognizer.session_stopped.connect(canceled_cb)

        # Start recognition
        recognizer.start_continuous_recognition_async()

        # Feed audio stream
        async def feed_audio():
            try:
                async for chunk in audio_stream:
                    stream.write(chunk)
            finally:
                stream.close()

        feed_task = asyncio.create_task(feed_audio())

        # Yield results
        try:
            while True:
                result = await result_queue.get()
                if result is None:
                    break
                yield result
        finally:
            recognizer.stop_continuous_recognition_async()
            await feed_task

    async def identify_language(
        self,
        audio_stream: AsyncGenerator[bytes, None],
        candidate_languages: List[LanguageCode]
    ) -> LanguageIdentificationResult:
        """Identify spoken language from audio."""
        stream = speechsdk.audio.PushAudioInputStream()
        audio_config = speechsdk.audio.AudioConfig(stream=stream)

        # Create language detection config
        language_codes = [lang.value for lang in candidate_languages]
        auto_detect_config = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
            languages=language_codes
        )

        # Create recognizer
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=self._speech_config,
            audio_config=audio_config,
            auto_detect_source_language_config=auto_detect_config
        )

        # Feed audio and recognize
        async def feed_audio():
            async for chunk in audio_stream:
                stream.write(chunk)
            stream.close()

        feed_task = asyncio.create_task(feed_audio())

        # Get result (timeout after language detection window)
        timeout = self.settings.conversation.language_detection_timeout_seconds
        result = await asyncio.wait_for(
            asyncio.to_thread(recognizer.recognize_once),
            timeout=timeout
        )

        await feed_task

        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            detected_lang_str = result.properties.get(
                speechsdk.PropertyId.SpeechServiceConnection_AutoDetectSourceLanguageResult
            )
            detected_lang = LanguageCode(detected_lang_str) if detected_lang_str else candidate_languages[0]

            return LanguageIdentificationResult(
                language=detected_lang,
                confidence=0.9  # Azure doesn't provide confidence, assume high
            )
        else:
            # Default to first candidate
            return LanguageIdentificationResult(
                language=candidate_languages[0],
                confidence=0.5
            )

    async def synthesize(
        self,
        text: str,
        language: LanguageCode,
        voice_name: Optional[str] = None
    ) -> AsyncGenerator[TTSAudioChunk, None]:
        """Text-to-speech synthesis."""
        # Configure voice
        voice = voice_name or self._get_voice_name(language)
        self._speech_config.speech_synthesis_voice_name = voice

        # Use pull stream for better control
        stream_callback = speechsdk.audio.PullAudioOutputStream()
        audio_config = speechsdk.audio.AudioOutputConfig(stream=stream_callback)

        synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=self._speech_config,
            audio_config=audio_config
        )

        # Create SSML
        ssml = f"""
        <speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='{language.value}'>
            <voice name='{voice}'>{text}</voice>
        </speak>
        """

        # Start synthesis
        result = await asyncio.to_thread(synthesizer.speak_ssml_async(ssml).get)

        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            # Stream audio in chunks
            audio_data = result.audio_data
            chunk_size = 3200  # 100ms at 16kHz mono PCM16
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i + chunk_size]
                yield TTSAudioChunk(
                    audio_data=chunk,
                    is_final=(i + chunk_size >= len(audio_data))
                )
        else:
            raise Exception(f"Synthesis failed: {result.reason}")

    async def cancel_synthesis(self, generation_id: str) -> None:
        """Cancel ongoing synthesis."""
        if generation_id in self._active_synthesizers:
            synth = self._active_synthesizers.pop(generation_id)
            synth.stop_speaking_async()


class FakeSpeechProvider(SpeechProvider):
    """Fake speech provider for testing."""

    async def recognize_continuous(
        self,
        audio_stream: AsyncGenerator[bytes, None],
        language: Optional[LanguageCode] = None
    ) -> AsyncGenerator[SpeechRecognitionResult, None]:
        """Echo back fake transcription."""
        # Drain audio stream
        async for _ in audio_stream:
            pass

        yield SpeechRecognitionResult(
            text="[Fake transcription] This is a test query",
            is_final=True,
            confidence=0.95,
            language=language or LanguageCode.EN_IN,
            event_type=SpeechEvent.RECOGNIZED
        )

    async def identify_language(
        self,
        audio_stream: AsyncGenerator[bytes, None],
        candidate_languages: List[LanguageCode]
    ) -> LanguageIdentificationResult:
        """Return first candidate language."""
        async for _ in audio_stream:
            pass

        return LanguageIdentificationResult(
            language=candidate_languages[0],
            confidence=0.95
        )

    async def synthesize(
        self,
        text: str,
        language: LanguageCode,
        voice_name: Optional[str] = None
    ) -> AsyncGenerator[TTSAudioChunk, None]:
        """Generate fake audio."""
        # Generate silent PCM16 audio
        duration_ms = len(text) * 50  # 50ms per character
        samples = int(16000 * duration_ms / 1000)
        fake_audio = b'\x00\x00' * samples

        # Yield in chunks
        chunk_size = 3200
        for i in range(0, len(fake_audio), chunk_size):
            chunk = fake_audio[i:i + chunk_size]
            await asyncio.sleep(0.1)  # Simulate streaming
            yield TTSAudioChunk(
                audio_data=chunk,
                is_final=(i + chunk_size >= len(fake_audio))
            )

    async def cancel_synthesis(self, generation_id: str) -> None:
        """No-op for fake provider."""
        pass
