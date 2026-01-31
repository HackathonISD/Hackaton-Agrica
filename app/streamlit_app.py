"""
Interface utilisateur Streamlit pour le chatbot AGRICA.
"""

import sys
import os
import base64
import glob
import time
import io

# Ajouter le dossier utils au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "utils"))

import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
from audio_recorder_streamlit import audio_recorder
from workflow import ConversationWorkflow
from voice_assistant import get_voice_assistant


# Charger le VoiceAssistant (en cache)
@st.cache_resource
def load_voice_assistant():
    """Charge le VoiceAssistant avec Whisper (STT) et Edge TTS (voix naturelle)."""
    return get_voice_assistant(
        stt_model="openai/whisper-small",  # Modèle STT multilingue
        default_language="fr",
        voice_gender="female",  # Voix féminine naturelle (Denise)
    )


# Chemins des dossiers de données
DATA_FOLDER = os.path.join(os.path.dirname(__file__), "..", "data")
PDF_FOLDERS = [
    os.path.join(
        DATA_FOLDER, "Corpus_Offres-Produits_AGRICA", "Corpus_Offres-Produits_AGRICA"
    ),
    os.path.join(
        DATA_FOLDER,
        "Complement_Corpus_Offres-Produits_AGRICA",
        "Complement_Corpus_Offres-Produits_AGRICA",
    ),
]


def stream_response(response: str):
    """Générateur pour afficher la réponse en mode streaming."""
    # Diviser par mots pour un effet plus naturel
    words = response.split(" ")
    for i, word in enumerate(words):
        yield word + (" " if i < len(words) - 1 else "")
        time.sleep(0.02)  # Délai entre chaque mot


def speech_to_text(audio_bytes: bytes) -> tuple[str, str]:
    """Convertit l'audio en texte avec le VoiceAssistant (Whisper HuggingFace).

    Returns:
        Tuple (texte transcrit, langue détectée) ou (None, None) en cas d'erreur
    """
    try:
        voice_assistant = load_voice_assistant()
        return voice_assistant.speech_to_text(
            audio_bytes, language=None, detect_language=True
        )
    except Exception as e:
        st.error(f"Erreur de transcription: {e}")
        return None, None


def detect_text_language(text: str) -> str:
    """Détecte la langue d'un texte en utilisant des patterns simples.

    Returns:
        Code de langue (fr, en, es, de, it, pt, ar, etc.) ou 'fr' par défaut
    """
    text_lower = text.lower()

    # Patterns pour chaque langue (mots communs)
    patterns = {
        "fr": [
            "bonjour",
            "merci",
            "comment",
            "pourquoi",
            "quoi",
            "quel",
            "quelle",
            "est-ce",
            "je",
            "vous",
            "nous",
            "les",
            "des",
            "une",
            "pour",
            "avec",
            "dans",
            "sur",
            "que",
            "qui",
        ],
        "en": [
            "hello",
            "thank",
            "how",
            "why",
            "what",
            "which",
            "is",
            "are",
            "you",
            "we",
            "the",
            "for",
            "with",
            "this",
            "that",
            "have",
            "has",
            "can",
            "would",
            "could",
        ],
        "es": [
            "hola",
            "gracias",
            "cómo",
            "por qué",
            "qué",
            "cuál",
            "es",
            "son",
            "usted",
            "nosotros",
            "los",
            "las",
            "para",
            "con",
            "este",
            "esta",
            "tiene",
            "puede",
        ],
        "de": [
            "hallo",
            "danke",
            "wie",
            "warum",
            "was",
            "welche",
            "ist",
            "sind",
            "sie",
            "wir",
            "die",
            "der",
            "das",
            "für",
            "mit",
            "haben",
            "kann",
            "können",
        ],
        "it": [
            "ciao",
            "grazie",
            "come",
            "perché",
            "cosa",
            "quale",
            "è",
            "sono",
            "lei",
            "noi",
            "il",
            "la",
            "per",
            "con",
            "questo",
            "questa",
            "ha",
            "può",
        ],
        "pt": [
            "olá",
            "obrigado",
            "como",
            "por que",
            "o que",
            "qual",
            "é",
            "são",
            "você",
            "nós",
            "os",
            "as",
            "para",
            "com",
            "este",
            "esta",
            "tem",
            "pode",
        ],
        "ar": ["مرحبا", "شكرا", "كيف", "لماذا", "ماذا", "أي", "هو", "هي", "أنت", "نحن"],
    }

    scores = {lang: 0 for lang in patterns}

    for lang, words in patterns.items():
        for word in words:
            if word in text_lower:
                scores[lang] += 1

    # Retourner la langue avec le score le plus élevé, ou 'fr' par défaut
    max_score = max(scores.values())
    if max_score > 0:
        return max(scores, key=scores.get)
    return "fr"


def text_to_speech(text: str, language: str = "fr") -> bytes:
    """Convertit le texte en audio avec le VoiceAssistant (Edge TTS - voix naturelle)."""
    try:
        voice_assistant = load_voice_assistant()
        return voice_assistant.text_to_speech(
            text, language=language, output_format="mp3"
        )
    except Exception as e:
        st.error(f"Erreur TTS: {e}")
        return None


def find_pdf_path(filename: str) -> str | None:
    """Recherche le chemin complet d'un fichier PDF."""
    # Nettoyer le nom de fichier
    clean_name = filename.strip()
    if not clean_name.lower().endswith(".pdf"):
        clean_name = clean_name + ".pdf"

    # Rechercher dans tous les dossiers PDF
    for folder in PDF_FOLDERS:
        if os.path.exists(folder):
            # Recherche récursive
            for root, dirs, files in os.walk(folder):
                for file in files:
                    if file.lower() == clean_name.lower():
                        return os.path.join(root, file)
                    # Aussi chercher si le nom est contenu dans le fichier
                    if clean_name.lower().replace(".pdf", "") in file.lower():
                        return os.path.join(root, file)
    return None


def display_pdf(file_path: str):
    """Affiche un PDF dans Streamlit avec streamlit-pdf-viewer."""
    try:
        with open(file_path, "rb") as f:
            pdf_bytes = f.read()

        # Utiliser streamlit-pdf-viewer pour un affichage fiable
        pdf_viewer(
            input=pdf_bytes,
            width=700,
            height=800,
        )

        # Ajouter un bouton de téléchargement comme option
        st.download_button(
            label="📥 Télécharger le PDF",
            data=pdf_bytes,
            file_name=os.path.basename(file_path),
            mime="application/pdf",
            use_container_width=True,
        )

    except FileNotFoundError:
        st.error(f"Fichier non trouvé: {file_path}")
    except Exception as e:
        st.error(f"Erreur lors de l'affichage du PDF: {str(e)}")


# Configuration de la page
st.set_page_config(
    page_title="Assistant AGRICA",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS personnalisé
st.markdown(
    """
    <style>
    /* Cacher le loader Streamlit (vélo/running) */
    [data-testid="stStatusWidget"] {
        display: none !important;
    }
    .stSpinner > div {
        display: none !important;
    }
    div[data-testid="stAppViewBlockContainer"] > div:first-child > div[data-testid="stVerticalBlock"] > div[data-stale="true"] {
        display: none !important;
    }
    
    /* Plein écran - supprimer les marges */
    .stApp {
        max-width: 100%;
        margin: 0;
        padding: 0;
    }
    
    /* Réduire le padding du contenu principal */
    .main .block-container {
        max-width: 100%;
        padding-left: 2rem;
        padding-right: 2rem;
        padding-top: 1rem;
    }
    
    /* Sidebar plus large */
    [data-testid="stSidebar"] {
        min-width: 300px;
    }
    
    /* Réduire la taille des titres dans le chat */
    [data-testid="stChatMessage"] h1 {
        font-size: 1.4rem !important;
        margin-top: 0.8rem;
        margin-bottom: 0.5rem;
    }
    [data-testid="stChatMessage"] h2 {
        font-size: 1.2rem !important;
        margin-top: 0.7rem;
        margin-bottom: 0.4rem;
    }
    [data-testid="stChatMessage"] h3 {
        font-size: 1.1rem !important;
        margin-top: 0.6rem;
        margin-bottom: 0.3rem;
    }
    [data-testid="stChatMessage"] h4 {
        font-size: 1rem !important;
        margin-top: 0.5rem;
        margin-bottom: 0.3rem;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #1976d2;
    }
    .assistant-message {
        background-color: #f5f5f5;
        border-left: 4px solid #4caf50;
    }
    .source-info {
        font-size: 0.85rem;
        color: #666;
        margin-top: 0.5rem;
        padding: 0.5rem;
        background-color: #fafafa;
        border-radius: 0.25rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_session_state():
    """Initialise les variables de session."""
    if "conversation" not in st.session_state:
        st.session_state.conversation = ConversationWorkflow(top_k=10)
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "selected_files_history" not in st.session_state:
        st.session_state.selected_files_history = []
    if "current_pdf" not in st.session_state:
        st.session_state.current_pdf = None
    if "show_pdf" not in st.session_state:
        st.session_state.show_pdf = False
    if "last_audio" not in st.session_state:
        st.session_state.last_audio = None
    if "voice_input" not in st.session_state:
        st.session_state.voice_input = None
    if "is_voice_question" not in st.session_state:
        st.session_state.is_voice_question = False
    if "autoplay_audio" not in st.session_state:
        st.session_state.autoplay_audio = False
    if "detected_language" not in st.session_state:
        st.session_state.detected_language = "fr"


def clear_conversation():
    """Efface l'historique de conversation."""
    st.session_state.conversation.clear_history()
    st.session_state.messages = []
    st.session_state.selected_files_history = []
    st.session_state.current_pdf = None
    st.session_state.show_pdf = False


def select_pdf(filename: str):
    """Sélectionne un PDF à afficher."""
    pdf_path = find_pdf_path(filename)
    if pdf_path:
        st.session_state.current_pdf = pdf_path
        st.session_state.show_pdf = True
    else:
        st.warning(f"Fichier non trouvé: {filename}")


def main():
    """Fonction principale de l'application."""
    init_session_state()

    # Sidebar
    with st.sidebar:
        st.image(
            "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTJCOgwI-twM6MtOVAyHZa30Llk9d3HeRWyWQ&s",
            width=200,
        )
        st.title("🌾 Assistant AGRICA")
        st.markdown("---")

        st.markdown(
            """
            ### À propos
            Cet assistant vous aide à comprendre les offres et produits 
            de protection sociale agricole d'AGRICA.
            
            Posez vos questions sur :
            - La santé et prévoyance
            - Les cotisations
            - Les garanties
            - Les régimes d'adhésion
            """
        )

        st.markdown("---")

        # Bouton pour effacer l'historique
        if st.button("🗑️ Nouvelle conversation", use_container_width=True):
            clear_conversation()
            st.rerun()

        # Afficher les fichiers utilisés dans la dernière réponse
        if st.session_state.selected_files_history:
            st.markdown("---")
            st.markdown("### 📁 Documents sources")
            st.markdown("*Cliquez pour afficher le PDF*")

            for file in st.session_state.selected_files_history[:5]:
                # Bouton pour chaque fichier
                if st.button(f"📄 {file}", key=f"pdf_{file}", use_container_width=True):
                    select_pdf(file)
                    st.rerun()

        # Bouton pour fermer le PDF
        if st.session_state.show_pdf:
            st.markdown("---")
            if st.button("❌ Fermer le PDF", use_container_width=True):
                st.session_state.show_pdf = False
                st.session_state.current_pdf = None
                st.rerun()

    # Layout principal avec colonnes
    if st.session_state.show_pdf and st.session_state.current_pdf:
        # Mode split: Chat + PDF
        col_chat, col_pdf = st.columns([1, 1])

        with col_chat:
            render_chat_interface()

        with col_pdf:
            st.markdown("### 📄 Document source")
            pdf_name = os.path.basename(st.session_state.current_pdf)
            st.caption(pdf_name)
            display_pdf(st.session_state.current_pdf)
    else:
        # Mode normal: Chat seul
        render_chat_interface()


def render_chat_interface():
    """Affiche l'interface de chat."""
    # Afficher l'historique des messages
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(
            message["role"], avatar="🧑" if message["role"] == "user" else "🤖"
        ):
            st.markdown(message["content"])
            # Ajouter le bouton audio pour les réponses de l'assistant
            if message["role"] == "assistant" and "audio" in message:
                # Autoplay seulement pour le dernier message si flag activé
                is_last_message = i == len(st.session_state.messages) - 1
                should_autoplay = is_last_message and st.session_state.get(
                    "autoplay_audio", False
                )

                if should_autoplay:
                    st.audio(message["audio"], format="audio/mp3", autoplay=True)
                    st.session_state.autoplay_audio = False  # Réinitialiser le flag
                else:
                    st.audio(message["audio"], format="audio/mp3")

    # Container fixe en bas pour l'entrée
    st.markdown(
        """
        <style>
        .input-container {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 10px 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Zone de saisie texte (en premier, pleine largeur)
    prompt = st.chat_input("Votre question sur la protection sociale agricole...")

    # Bouton microphone dans la sidebar
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🎤 Mode vocal")
        audio_bytes = audio_recorder(
            text="Cliquez pour parler",
            recording_color="#e74c3c",
            neutral_color="#3498db",
            icon_size="2x",
            pause_threshold=2.5,
        )

        # Traiter l'audio enregistré
        if audio_bytes and audio_bytes != st.session_state.get("last_audio"):
            st.session_state.last_audio = audio_bytes
            with st.spinner("🎤 Transcription avec Whisper..."):
                transcribed_text, detected_lang = speech_to_text(audio_bytes)
                if transcribed_text:
                    lang_names = {
                        "fr": "🇫🇷 Français",
                        "en": "🇬🇧 English",
                        "es": "🇪🇸 Español",
                        "de": "🇩🇪 Deutsch",
                        "it": "🇮🇹 Italiano",
                        "pt": "🇵🇹 Português",
                        "ar": "🇸🇦 العربية",
                    }
                    lang_display = lang_names.get(detected_lang, detected_lang)
                    st.success(f'✅ [{lang_display}] "{transcribed_text}"')
                    st.session_state.voice_input = transcribed_text
                    st.session_state.detected_language = detected_lang or "fr"
                    st.session_state.is_voice_question = (
                        True  # Marquer comme question vocale
                    )
                    time.sleep(1)
                    st.rerun()
                else:
                    st.warning("Je n'ai pas pu comprendre l'audio.")

    # Utiliser l'entrée vocale si disponible
    is_voice_question = False
    current_language = "fr"  # Langue par défaut

    if "voice_input" in st.session_state and st.session_state.voice_input:
        prompt = st.session_state.voice_input
        st.session_state.voice_input = None
        is_voice_question = st.session_state.get("is_voice_question", False)
        st.session_state.is_voice_question = False  # Réinitialiser
        current_language = st.session_state.get("detected_language", "fr")
    elif prompt:
        # Détecter la langue pour les questions texte
        current_language = detect_text_language(prompt)
        st.session_state.detected_language = current_language

    if prompt:
        # Fermer automatiquement le PDF affiché quand une nouvelle question est posée
        st.session_state.show_pdf = False
        st.session_state.current_pdf = None

        # Ajouter le message utilisateur
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)

        # Générer la réponse
        with st.chat_message("assistant", avatar="🤖"):
            with st.status("🔍 Recherche en cours...", expanded=True) as status:
                try:
                    # Passer la langue détectée au workflow
                    result = st.session_state.conversation.chat(
                        prompt, language=current_language
                    )
                    response = result["response"]

                    # Extraire les fichiers uniques des chunks similaires
                    similar_chunks = result.get("similar_chunks", [])
                    source_files = []
                    seen = set()
                    for chunk in similar_chunks:
                        fichier = chunk.get("fichier_nom", "")
                        if fichier and fichier not in seen:
                            seen.add(fichier)
                            source_files.append(fichier)

                    # Mettre à jour les fichiers sources
                    st.session_state.selected_files_history = source_files

                    # Afficher la réponse en mode streaming
                    st.write_stream(stream_response(response))

                    # Générer et jouer l'audio seulement si la question était vocale
                    if is_voice_question:
                        # Utiliser la langue détectée pour la réponse vocale
                        response_lang = st.session_state.get("detected_language", "fr")
                        with st.spinner(
                            f"🔊 Génération de la réponse vocale ({response_lang})..."
                        ):
                            audio_response = text_to_speech(
                                response, language=response_lang
                            )

                        if audio_response:
                            # Ajouter à l'historique avec l'audio
                            st.session_state.messages.append(
                                {
                                    "role": "assistant",
                                    "content": response,
                                    "audio": audio_response,
                                }
                            )
                            # Activer le flag pour autoplay après le rerun
                            st.session_state.autoplay_audio = True
                        else:
                            # Ajouter à l'historique sans audio
                            st.session_state.messages.append(
                                {"role": "assistant", "content": response}
                            )
                    else:
                        # Question textuelle: pas d'audio automatique
                        st.session_state.messages.append(
                            {"role": "assistant", "content": response}
                        )

                    # Forcer le rerun pour afficher les fichiers dans la sidebar
                    st.rerun()

                except Exception as e:
                    error_msg = f"Une erreur s'est produite : {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_msg}
                    )

    # Message d'accueil si pas de messages
    if not st.session_state.messages:
        st.markdown(
            """
            <div style="text-align: center; padding: 2rem; color: #666;">
                <h3>👋 Bienvenue !</h3>
                <p>Je suis votre assistant pour la protection sociale agricole.</p>
                <p>Posez-moi vos questions sur les offres, cotisations, garanties...</p>
                <br>
                <p><strong>Exemples de questions :</strong></p>
                <ul style="text-align: left; display: inline-block;">
                    <li>Quelles sont les garanties santé proposées ?</li>
                    <li>Comment fonctionne la prévoyance pour les salariés agricoles ?</li>
                    <li>Quel est le montant des cotisations retraite ?</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
