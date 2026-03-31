# 🌾 Assistant AGRICA - Assistant Protection Sociale Agricole

Un assistant conversationnel intelligent pour la protection sociale agricole, utilisant un système RAG (Retrieval-Augmented Generation) avec support vocal multilingue.

## Demo video

Please refer to this link: https://drive.google.com/file/d/1_puYyd9DZbZ23thrrI_MHZ32MM0ZMniz/view?usp=sharing

## ✨ Fonctionnalités

- 💬 **Chat intelligent** : Posez des questions sur la protection sociale agricole (santé, prévoyance, cotisations, garanties)
- 🔍 **Recherche sémantique** : Recherche dans une base de documents vectorisée avec FAISS
- 🎤 **Mode vocal** : Reconnaissance vocale avec Whisper et synthèse vocale avec Edge TTS
- 🌍 **Multilingue** : Détection automatique de la langue et réponse dans la langue de l'utilisateur (FR, EN, ES, DE, IT, PT, AR...)
- 📄 **Affichage PDF** : Visualisation des documents sources directement dans l'interface

## 🏗️ Architecture

```
Hackaton/
├── app/
│   ├── streamlit_app.py      # Interface utilisateur Streamlit
│   └── utils/
│       ├── workflow.py       # Workflow LangGraph avec mémoire
│       ├── llm.py            # Client OpenAI/Snowflake Cortex
│       ├── voice_assistant.py # STT (Whisper) + TTS (Edge TTS)
│       ├── embedde_document.py # Embeddings avec sentence-transformers
│       ├── create_vector_database.py # Création base vectorielle FAISS
│       ├── retrive_similar_documents.py # Recherche similaire
│       └── select_relevant_files.py # Sélection documents pertinents
├── data/
│   ├── files_index.csv       # Index des documents avec résumés
│   ├── vector_database/      # Base FAISS + métadonnées
│   └── markdown_output/      # Documents convertis en markdown
├── requirements.txt          # Dépendances Python
├── setup_models.py           # Script de téléchargement des modèles
└── .env.example              # Template variables d'environnement
```

## 🚀 Installation

### Prérequis

- Python 3.9 ou supérieur
- pip

### 1. Cloner le projet

```bash
git clone <url-du-repo>
cd Hackaton
```

### 2. Créer un environnement virtuel

**Windows (PowerShell) :**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux :**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Configurer les variables d'environnement

Copier le fichier d'exemple et le modifier :

```bash
cp .env.example .env
```

Éditer `.env` avec vos credentials :
```dotenv
OPENAI_TOKEN=votre_token_api
PROVIDER_URL=https://votre-provider.com/api/v2/cortex/v1
```

> **Note** : Ce projet utilise Snowflake Cortex comme provider LLM. Remplacez par vos propres credentials.

### 5. Télécharger les modèles ML

Exécuter le script de setup pour pré-télécharger les modèles (embedding et Whisper) :

```bash
python setup_models.py
```

Cette étape télécharge :
- 📦 **intfloat/multilingual-e5-base** (~1.1 GB) - Modèle d'embedding multilingue
- 📦 **openai/whisper-small** (~967 MB) - Modèle de reconnaissance vocale

### 6. Lancer l'application

```bash
streamlit run app/streamlit_app.py
```

L'application sera accessible sur : http://localhost:8501

## 📖 Utilisation

### Mode Texte
1. Tapez votre question dans le champ de saisie
2. La langue est détectée automatiquement
3. La réponse est générée dans la même langue

### Mode Vocal
1. Cliquez sur le bouton microphone 🎤 dans la sidebar
2. Parlez votre question (dans n'importe quelle langue supportée)
3. La transcription s'affiche avec la langue détectée
4. La réponse est générée et lue à voix haute dans votre langue

### Langues supportées
- 🇫🇷 Français
- 🇬🇧 English
- 🇪🇸 Español
- 🇩🇪 Deutsch
- 🇮🇹 Italiano
- 🇵🇹 Português
- 🇸🇦 العربية

## 🛠️ Technologies utilisées

| Composant | Technologie |
|-----------|-------------|
| Interface | Streamlit |
| LLM | OpenAI API / Snowflake Cortex |
| Orchestration | LangGraph + LangChain |
| Embeddings | sentence-transformers (multilingual-e5-base) |
| Base vectorielle | FAISS |
| STT (Speech-to-Text) | Whisper (HuggingFace Transformers) |
| TTS (Text-to-Speech) | Edge TTS (Microsoft Azure) |
| Conversion PDF | Docling |

## 🐳 Docker

Build et lancer avec Docker :

```bash
docker build -t agrica-assistant .
docker run -p 8501:8501 --env-file .env agrica-assistant
```

## ⚠️ Troubleshooting

### Erreur 401 - Token invalide
Vérifiez que votre `OPENAI_TOKEN` dans `.env` est valide et non expiré.

### Modèles non téléchargés
Relancez `python setup_models.py` pour télécharger les modèles.

### Erreur d'activation PowerShell
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Problème de mémoire
Les modèles Whisper nécessitent ~2GB de RAM. Utilisez `whisper-tiny` si nécessaire.

## 👥 Équipe

Projet réalisé dans le cadre du Hackathon Hack'in Saclay - M2 ISD Paris Saclay

## 📝 License

MIT License
