"""
Module de reconnaissance vocale (STT) et synthèse vocale (TTS)
utilisant Whisper (Hugging Face) et Edge TTS (Microsoft Azure).

Modèles utilisés:
- STT: openai/whisper-small (multilingue, excellent pour le français)
- TTS: Edge TTS (Microsoft Azure) - voix naturelles haute qualité
"""

import io
import os
import re
import tempfile
import asyncio
from typing import Optional, Literal
import numpy as np

import torch
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
)
import soundfile as sf
import edge_tts
from num2words import num2words


class VoiceAssistant:
    """
    Classe pour la reconnaissance vocale (STT) et la synthèse vocale (TTS)
    utilisant Whisper (Hugging Face) et Edge TTS (Microsoft Azure).
    """

    # Langues supportées pour Whisper
    SUPPORTED_LANGUAGES = {
        "fr": "french",
        "en": "english",
        "es": "spanish",
        "de": "german",
        "it": "italian",
        "pt": "portuguese",
        "nl": "dutch",
        "pl": "polish",
        "ru": "russian",
        "zh": "chinese",
        "ja": "japanese",
        "ko": "korean",
        "ar": "arabic",
    }

    # Voix Edge TTS par langue (voix naturelles Microsoft Azure)
    EDGE_TTS_VOICES = {
        "fr": "fr-FR-DeniseNeural",  # Voix féminine française naturelle
        "fr-m": "fr-FR-HenriNeural",  # Voix masculine française
        "en": "en-US-JennyNeural",  # Voix féminine anglaise
        "en-m": "en-US-GuyNeural",  # Voix masculine anglaise
        "es": "es-ES-ElviraNeural",  # Voix féminine espagnole
        "de": "de-DE-KatjaNeural",  # Voix féminine allemande
        "it": "it-IT-ElsaNeural",  # Voix féminine italienne
        "pt": "pt-BR-FranciscaNeural",  # Voix féminine portugaise
        "ar": "ar-SA-ZariyahNeural",  # Voix féminine arabe
    }

    def __init__(
        self,
        stt_model: str = "openai/whisper-small",
        default_language: str = "fr",
        device: Optional[str] = None,
        voice_gender: str = "female",
    ):
        """
        Initialise le VoiceAssistant.

        Args:
            stt_model: Modèle Whisper à utiliser (tiny, base, small, medium, large)
            default_language: Langue par défaut (code ISO)
            device: Device à utiliser (cuda, cpu, ou auto)
            voice_gender: Genre de la voix ("female" ou "male")
        """
        self.default_language = default_language
        self.voice_gender = voice_gender

        # Déterminer le device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"🎤 Initialisation VoiceAssistant sur {self.device}...")

        # Initialiser le modèle STT (Whisper)
        self._init_stt(stt_model)

        # Configurer la voix TTS
        self._set_voice(default_language, voice_gender)

        print("✅ VoiceAssistant prêt!")

    def _init_stt(self, model_name: str):
        """Initialise le modèle Speech-to-Text (Whisper)."""
        print(f"  📥 Chargement du modèle STT: {model_name}")
        self.stt_processor = WhisperProcessor.from_pretrained(model_name)
        self.stt_model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.stt_model.to(self.device)
        self.stt_model.eval()

    def _set_voice(self, language: str, gender: str = "female"):
        """Configure la voix Edge TTS."""
        if gender == "male" and f"{language}-m" in self.EDGE_TTS_VOICES:
            self.current_voice = self.EDGE_TTS_VOICES[f"{language}-m"]
        else:
            self.current_voice = self.EDGE_TTS_VOICES.get(
                language, self.EDGE_TTS_VOICES["fr"]
            )
        print(f"  🔊 Voix TTS configurée: {self.current_voice}")

    def speech_to_text(
        self,
        audio_bytes: bytes,
        language: Optional[str] = None,
        detect_language: bool = False,
    ) -> Optional[str] | tuple[Optional[str], Optional[str]]:
        """
        Convertit un audio en texte avec Whisper.

        Args:
            audio_bytes: Données audio en bytes (WAV)
            language: Code de langue (fr, en, etc.) ou None pour auto-détection
            detect_language: Si True, retourne aussi la langue détectée

        Returns:
            Si detect_language=False: Texte transcrit ou None en cas d'erreur
            Si detect_language=True: Tuple (texte transcrit, langue détectée) ou (None, None)
        """
        try:
            # Sauvegarder temporairement l'audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

            # Charger l'audio
            audio_data, sr = sf.read(tmp_path)

            # Convertir en mono si stéréo
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)

            # Rééchantillonner si nécessaire
            if sr != 16000:
                import scipy.signal as signal

                audio_data = signal.resample(
                    audio_data, int(len(audio_data) * 16000 / sr)
                )

            # Nettoyer le fichier temporaire
            os.unlink(tmp_path)

            # Préparer l'audio pour Whisper
            input_features = self.stt_processor(
                audio_data, sampling_rate=16000, return_tensors="pt"
            ).input_features.to(self.device)

            # Détecter la langue si demandé
            detected_lang = None
            if detect_language or language is None:
                with torch.no_grad():
                    # Utiliser la méthode native de Whisper pour détecter la langue
                    # C'est plus fiable que d'analyser les logits manuellement
                    predicted_ids_for_lang = self.stt_model.generate(
                        input_features,
                        max_new_tokens=1,  # Juste le token de langue
                        return_dict_in_generate=True,
                        output_scores=True,
                    )

                    # Extraire le token de langue des séquences générées
                    generated_tokens = predicted_ids_for_lang.sequences[0]

                    # Chercher le token de langue dans les premiers tokens générés
                    for token_id in generated_tokens[
                        1:5
                    ]:  # Ignorer le premier (start token)
                        token = self.stt_processor.tokenizer.decode([token_id.item()])
                        if token.startswith("<|") and token.endswith("|>"):
                            potential_lang = token.replace("<|", "").replace("|>", "")
                            # Vérifier si c'est un code de langue valide (2-3 caractères)
                            if (
                                len(potential_lang) in [2, 3]
                                and potential_lang.isalpha()
                            ):
                                detected_lang = potential_lang
                                break

                    # Fallback: utiliser l'ancienne méthode si la nouvelle échoue
                    if detected_lang is None:
                        decoder_input_ids = torch.tensor(
                            [[self.stt_model.config.decoder_start_token_id]]
                        ).to(self.device)
                        logits = self.stt_model(
                            input_features, decoder_input_ids=decoder_input_ids
                        ).logits

                        # Les tokens de langue sont dans une plage spécifique
                        lang_tokens = (
                            self.stt_processor.tokenizer.additional_special_tokens
                        )
                        lang_token_ids = [
                            self.stt_processor.tokenizer.convert_tokens_to_ids(t)
                            for t in lang_tokens
                            if t.startswith("<|")
                            and t.endswith("|>")
                            and 4 <= len(t) <= 7
                        ]

                        if lang_token_ids:
                            lang_logits = logits[0, 0, lang_token_ids]
                            predicted_lang_idx = lang_logits.argmax().item()
                            predicted_lang_token = (
                                self.stt_processor.tokenizer.convert_ids_to_tokens(
                                    lang_token_ids[predicted_lang_idx]
                                )
                            )
                            detected_lang = predicted_lang_token.replace(
                                "<|", ""
                            ).replace("|>", "")

                    print(f"  🌍 Langue détectée: {detected_lang}")

            # Configurer la langue pour forcer la transcription (pas la traduction)
            # Utiliser la langue détectée si aucune langue n'est spécifiée
            forced_decoder_ids = None
            lang_to_use = language or detected_lang or self.default_language
            lang_name = self.SUPPORTED_LANGUAGES.get(lang_to_use, lang_to_use)
            forced_decoder_ids = self.stt_processor.get_decoder_prompt_ids(
                language=lang_name, task="transcribe"  # IMPORTANT: transcribe, pas translate
            )

            # Générer la transcription dans la langue d'origine
            with torch.no_grad():
                predicted_ids = self.stt_model.generate(
                    input_features,
                    forced_decoder_ids=forced_decoder_ids,
                    max_length=448,
                )

            # Décoder le texte
            transcription = self.stt_processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0]

            result_text = transcription.strip()

            if detect_language:
                return result_text, detected_lang or self.default_language
            return result_text

        except Exception as e:
            print(f"❌ Erreur STT: {e}")
            if detect_language:
                return None, None
            return None

    def text_to_speech(
        self,
        text: str,
        language: Optional[str] = None,
        output_format: Literal["wav", "mp3"] = "mp3",
    ) -> Optional[bytes]:
        """
        Convertit du texte en audio avec Edge TTS.

        Args:
            text: Texte à convertir
            language: Code de langue (pour changer la voix)
            output_format: Format de sortie (wav ou mp3)

        Returns:
            Données audio en bytes ou None en cas d'erreur
        """
        try:
            # Nettoyer et préparer le texte
            clean_text = self._clean_text_for_speech(
                text, language or self.default_language
            )

            if not clean_text:
                return None

            # Limiter la longueur
            if len(clean_text) > 3000:
                clean_text = clean_text[:3000] + "... La suite est affichée à l'écran."

            # Changer la voix si la langue est différente
            voice = self.current_voice
            if language and language != self.default_language:
                voice = self.EDGE_TTS_VOICES.get(language, self.current_voice)

            # Générer l'audio avec Edge TTS (async)
            audio_bytes = self._run_edge_tts(clean_text, voice)

            return audio_bytes

        except Exception as e:
            print(f"❌ Erreur TTS: {e}")
            return None

    def _run_edge_tts(self, text: str, voice: str) -> Optional[bytes]:
        """Exécute Edge TTS de manière synchrone."""
        try:
            # Créer une nouvelle boucle event pour l'exécution async
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self._generate_edge_tts(text, voice))
            loop.close()
            return result
        except Exception as e:
            print(f"❌ Erreur Edge TTS: {e}")
            return None

    async def _generate_edge_tts(self, text: str, voice: str) -> Optional[bytes]:
        """Génère l'audio avec Edge TTS (async)."""
        communicate = edge_tts.Communicate(text, voice)
        audio_buffer = io.BytesIO()

        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_buffer.write(chunk["data"])

        audio_buffer.seek(0)
        return audio_buffer.read()

    def _clean_text_for_speech(self, text: str, language: str = "fr") -> str:
        """
        Nettoie et prépare le texte pour la synthèse vocale.
        Convertit les chiffres en mots pour une lecture naturelle.
        """
        # Enlever le markdown
        clean = text.replace("**", "").replace("*", "")
        clean = clean.replace("#", "").replace("`", "")
        clean = clean.replace("- ", ", ")
        clean = clean.replace("\n\n", ". ")
        clean = clean.replace("\n", " ")

        # Convertir les pourcentages (ex: 50% -> cinquante pour cent)
        clean = re.sub(
            r"(\d+(?:[.,]\d+)?)\s*%",
            lambda m: self._number_to_words(m.group(1), language) + " pour cent",
            clean,
        )

        # Convertir les montants en euros (ex: 100€ -> cent euros)
        clean = re.sub(
            r"(\d+(?:[.,]\d+)?)\s*€",
            lambda m: self._number_to_words(m.group(1), language) + " euros",
            clean,
        )

        # Convertir les montants en dollars (ex: $50 -> cinquante dollars)
        clean = re.sub(
            r"\$\s*(\d+(?:[.,]\d+)?)",
            lambda m: self._number_to_words(m.group(1), language) + " dollars",
            clean,
        )

        # Convertir les années (ex: 2024 -> deux mille vingt-quatre)
        # Les années sont des nombres à 4 chiffres commençant par 19 ou 20
        clean = re.sub(
            r"\b(19\d{2}|20\d{2})\b",
            lambda m: self._number_to_words(m.group(1), language),
            clean,
        )

        # Convertir les nombres décimaux avec virgule (ex: 3,5 -> trois virgule cinq)
        clean = re.sub(
            r"(\d+),(\d+)",
            lambda m: self._number_to_words(m.group(1), language)
            + " virgule "
            + self._number_to_words(m.group(2), language),
            clean,
        )

        # Convertir les nombres décimaux avec point (ex: 3.5 -> trois point cinq)
        clean = re.sub(
            r"(\d+)\.(\d+)",
            lambda m: self._number_to_words(m.group(1), language)
            + " point "
            + self._number_to_words(m.group(2), language),
            clean,
        )

        # Convertir les nombres entiers restants (ex: 42 -> quarante-deux)
        clean = re.sub(
            r"\b(\d+)\b", lambda m: self._number_to_words(m.group(1), language), clean
        )

        # Nettoyer les espaces multiples
        while "  " in clean:
            clean = clean.replace("  ", " ")

        return clean.strip()

    def _number_to_words(self, number_str: str, language: str = "fr") -> str:
        """Convertit un nombre en mots dans la langue spécifiée."""
        try:
            # Nettoyer le nombre (enlever espaces et remplacer virgule par point)
            number_str = number_str.strip().replace(",", ".").replace(" ", "")
            number = float(number_str)

            # Si c'est un entier, le convertir en int pour éviter "virgule zéro"
            if number.is_integer():
                number = int(number)

            return num2words(number, lang=language)
        except Exception:
            # En cas d'erreur, retourner le nombre original
            return number_str

    def change_voice(self, language: str = None, gender: str = None):
        """Change la voix TTS."""
        if language:
            self.default_language = language
        if gender:
            self.voice_gender = gender

        self._set_voice(language or self.default_language, gender or self.voice_gender)


# Singleton pour réutiliser le modèle chargé
_voice_assistant_instance: Optional[VoiceAssistant] = None


def get_voice_assistant(
    stt_model: str = "openai/whisper-small",
    default_language: str = "fr",
    voice_gender: str = "female",
) -> VoiceAssistant:
    """
    Obtient une instance singleton du VoiceAssistant.

    Args:
        stt_model: Modèle Whisper à utiliser
        default_language: Langue par défaut
        voice_gender: Genre de la voix ("female" ou "male")

    Returns:
        Instance de VoiceAssistant
    """
    global _voice_assistant_instance

    if _voice_assistant_instance is None:
        _voice_assistant_instance = VoiceAssistant(
            stt_model=stt_model,
            default_language=default_language,
            voice_gender=voice_gender,
        )

    return _voice_assistant_instance
