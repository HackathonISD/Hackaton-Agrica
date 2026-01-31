"""
Script de téléchargement des modèles nécessaires.
Exécuter ce script une fois avant de lancer l'application Streamlit.

Usage:
    python setup_models.py
"""

import os


def download_models():
    """Télécharge tous les modèles nécessaires pour l'application."""

    print("=" * 60)
    print("TÉLÉCHARGEMENT DES MODÈLES")
    print("=" * 60)

    # 1. Modèle d'embedding pour la recherche vectorielle
    print("\n[1/2] Téléchargement du modèle d'embedding (multilingual-e5-base)...")
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("intfloat/multilingual-e5-base")
        print("✅ Modèle d'embedding téléchargé avec succès!")
    except Exception as e:
        print(f"❌ Erreur lors du téléchargement du modèle d'embedding: {e}")

    # 2. Modèle Whisper pour la reconnaissance vocale (STT)
    print("\n[2/2] Téléchargement du modèle Whisper (openai/whisper-small)...")
    try:
        from transformers import WhisperProcessor, WhisperForConditionalGeneration

        processor = WhisperProcessor.from_pretrained("openai/whisper-small")
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
        print("✅ Modèle Whisper téléchargé avec succès!")
    except Exception as e:
        print(f"❌ Erreur lors du téléchargement du modèle Whisper: {e}")

    print("\n" + "=" * 60)
    print("TÉLÉCHARGEMENT TERMINÉ!")
    print("=" * 60)
    print("\nVous pouvez maintenant lancer l'application avec:")
    print("  streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    download_models()
