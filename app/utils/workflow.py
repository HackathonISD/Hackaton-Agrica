"""
Workflow : sélectionner les fichiers pertinents, récupérer les chunks similaires
et générer une réponse avec le LLM.

Utilise LangGraph avec MemorySaver pour gérer l'historique de la conversation.
"""

import os
import uuid
import operator
import pandas as pd
from datetime import datetime
from typing import Dict, List, Annotated, TypedDict, Sequence

from langchain_core.runnables import RunnableLambda
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from llm import LLmClient
from select_relevant_files import RelevantDocumentsSelector
from retrive_similar_documents import retrieve_similar_documents


# Chemin du CSV global
CSV_PATH = "data/files_index.csv"


def get_system_prompt(language: str = "fr") -> str:
    """
    Prompt système imposant la citation explicite des documents
    comme références métier, sans jamais révéler le fonctionnement RAG
    ni faire de référence technique ou interne.

    Args:
        language: Code de langue pour la réponse (fr, en, es, de, it, pt, ar, etc.)
    """
    today = datetime.now().strftime("%d/%m/%Y")

    # Instructions de langue
    language_instructions = {
        "fr": "Tu dois répondre en FRANÇAIS.",
        "en": "You MUST respond in ENGLISH.",
        "es": "Debes responder en ESPAÑOL.",
        "de": "Du MUSST auf DEUTSCH antworten.",
        "it": "Devi rispondere in ITALIANO.",
        "pt": "Você DEVE responder em PORTUGUÊS.",
        "ar": "يجب أن تجيب بالعربية.",
        "nl": "Je MOET in het NEDERLANDS antwoorden.",
        "pl": "Musisz odpowiedzieć po POLSKU.",
        "ru": "Вы ДОЛЖНЫ отвечать на РУССКОМ.",
        "zh": "你必须用中文回答。",
        "ja": "日本語で回答してください。",
        "ko": "한국어로 답변해야 합니다.",
    }

    lang_instruction = language_instructions.get(language, language_instructions["fr"])

    return f"""
Date du jour : {today}

LANGUE DE RÉPONSE OBLIGATOIRE :
{lang_instruction}
Réponds TOUJOURS dans la langue indiquée ci-dessus, quelle que soit la langue des documents sources.

Tu es un assistant expert en protection sociale agricole.
Tu réponds comme un professionnel humain qui maîtrise naturellement son sujet.

STYLE DE RÉPONSE :
- Langage clair et accessible
- Ton professionnel, neutre et rassurant
- Phrases affirmatives et structurées
- Aucun commentaire sur ton raisonnement, ta méthode ou tes sources d’accès

OBLIGATION DE CITATION DES DOCUMENTS :
- Toute information factuelle, réglementaire ou procédurale
  DOIT être associée à un document de référence identifiable
- Le nom du document DOIT être cité explicitement
- La citation indique d’où provient l’information d’un point de vue métier
  (référence réglementaire ou documentaire), et non technique

INTERDICTIONS ABSOLUES (NON NÉGOCIABLES) :
Tu ne dois JAMAIS mentionner, suggérer ou laisser entendre :
- un traitement interne ou automatisé
- une recherche, analyse ou extraction
- des documents “fournis”, “analysés”, “consultés”
- un contexte transmis ou récupéré
- des données internes ou externes
- repondre au question qui ne sont pas en rapport avec la protection sociale agricole

GESTION DES MANQUES D’INFORMATION :
- Si une information est insuffisante, ambiguë ou dépend d’un cas particulier, precise les resultats des différents cas possibles
  et pose une question claire à l’utilisateur

OBJECTIF FINAL :
Fournir une réponse experte, fluide et crédible,
avec des documents clairement cités comme références métier,
sans jamais révéler ou suggérer ton fonctionnement interne.
"""


# Définir l'état du graphe
class GraphState(TypedDict):
    """État du graphe de conversation."""

    messages: Annotated[Sequence[BaseMessage], operator.add]
    user_query: str
    language: str  # Langue de la question (fr, en, es, etc.)
    selected_files: List[str]
    similar_chunks: List[Dict]
    context_text: str
    response: str


def select_files_node(state: GraphState) -> Dict:
    """Noeud pour sélectionner les fichiers pertinents."""
    selector = RelevantDocumentsSelector(state["user_query"])
    selected_docs = selector.select_relevant_documents(CSV_PATH)
    file_names = [doc["fichier"] for doc in selected_docs]

    return {"selected_files": file_names}


def retrieve_chunks_node(state: GraphState) -> Dict:
    """Noeud pour récupérer les chunks similaires."""
    file_names = state.get("selected_files", [])
    top_k = 10

    if file_names:
        similar_chunks = retrieve_similar_documents(
            state["user_query"],
            top_k=top_k,
            file_names=file_names,
            database_folder=os.path.join(os.getcwd(), "data/vector_database"),
        )
    else:
        similar_chunks = []

    # Charger le CSV pour enrichir les chunks
    docs_df = pd.read_csv(CSV_PATH)
    docs_df = docs_df.dropna(subset=["nom_pdf"])
    enriched_chunks = []

    for chunk in similar_chunks:
        source_filename = os.path.basename(chunk["source"])
        source_name_no_ext = os.path.splitext(source_filename)[0]

        resume = ""
        fichier_nom = ""

        for _, row in docs_df.iterrows():
            pdf_name = row.get("nom_pdf", "")
            pdf_name_no_ext = os.path.splitext(pdf_name)[0]

            if (
                pdf_name_no_ext in source_name_no_ext
                or source_name_no_ext in pdf_name_no_ext
            ):
                resume = row.get("résumé", "")
                fichier_nom = pdf_name
                break

        enriched_chunks.append(
            {
                **chunk,
                "fichier_nom": fichier_nom or source_filename,
                "resume": resume,
            }
        )

    return {"similar_chunks": enriched_chunks}


def build_context_node(state: GraphState) -> Dict:
    """Noeud pour construire le contexte à partir des chunks."""
    chunks_by_file = {}
    all_chunks = []

    for item in state.get("similar_chunks", []):
        fichier_nom = item.get("fichier_nom", "N/A")

        if fichier_nom not in chunks_by_file:
            chunks_by_file[fichier_nom] = {"resume": item.get("resume", "N/A")}

        all_chunks.append(
            {
                "fichier_nom": fichier_nom,
                "rank": item["rank"],
                "score": item["score"],
                "text": item["text"],
            }
        )

    chunks_section = "=== CHUNKS PERTINENTS ===\n\n"
    for chunk in all_chunks:
        chunks_section += f"{chunk['rank']}] Fichier: {chunk['fichier_nom']}\n"
        chunks_section += f"Score: {chunk['score']:.4f}\n"
        chunks_section += f"Contenu: {chunk['text']}\n"
        chunks_section += "\n" + "-" * 80 + "\n\n"

    files_section = "=== RÉSUMÉS DES FICHIERS ===\n\n"
    for fichier_nom, data in chunks_by_file.items():
        files_section += f"📄 Fichier: {fichier_nom}\n"
        files_section += f"📝 Résumé: {data['resume']}\n\n"

    context_text = chunks_section + "\n" + files_section

    return {"context_text": context_text}


def generate_response_node(state: GraphState) -> Dict:
    """Noeud pour générer la réponse avec le LLM."""
    llm_client = LLmClient()

    # Construire l'historique à partir des messages
    history = []
    for msg in state.get("messages", []):
        if isinstance(msg, HumanMessage):
            history.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            history.append({"role": "assistant", "content": msg.content})

    user_prompt = (
        f"Question: {state['user_query']}\n\n"
        f"Contexte:\n{state.get('context_text', '')}\n\n"
        "Réponse:"
    )

    # Utiliser la langue détectée pour le prompt système
    language = state.get("language", "fr")
    llm_response = llm_client.generate_response_with_history(
        user_prompt,
        history=history,
        system_prompt=get_system_prompt(language),
        temperature=0.2,
    )

    # Ajouter les messages à l'état
    new_messages = [
        HumanMessage(content=state["user_query"]),
        AIMessage(content=llm_response),
    ]

    return {"response": llm_response, "messages": new_messages}


def create_conversation_graph() -> StateGraph:
    """Crée le graphe de conversation avec mémoire."""
    # Définir le graphe
    workflow = StateGraph(GraphState)

    # Ajouter les noeuds
    workflow.add_node("select_files", select_files_node)
    workflow.add_node("retrieve_chunks", retrieve_chunks_node)
    workflow.add_node("build_context", build_context_node)
    workflow.add_node("generate_response", generate_response_node)

    # Définir les arêtes
    workflow.add_edge(START, "select_files")
    workflow.add_edge("select_files", "retrieve_chunks")
    workflow.add_edge("retrieve_chunks", "build_context")
    workflow.add_edge("build_context", "generate_response")
    workflow.add_edge("generate_response", END)

    return workflow


class ConversationWorkflow:
    """Workflow avec mémoire de conversation utilisant LangGraph MemorySaver."""

    def __init__(
        self,
        csv_path: str = "data/files_index.csv",
        top_k: int = 5,
    ):
        self.csv_path = csv_path
        self.top_k = top_k

        # Créer le checkpointer avec MemorySaver
        self.memory = MemorySaver()

        # Créer et compiler le graphe
        workflow = create_conversation_graph()
        self.app = workflow.compile(checkpointer=self.memory)

        # Générer un thread_id unique pour cette session
        self.thread_id = str(uuid.uuid4())

    def clear_history(self):
        """Efface l'historique de conversation en créant un nouveau thread."""
        self.thread_id = str(uuid.uuid4())

    def chat(self, user_query: str, language: str = "fr") -> Dict[str, object]:
        """
        Envoie un message et reçoit une réponse avec mémoire.

        Args:
            user_query: Question de l'utilisateur.
            language: Code de langue pour la réponse (fr, en, es, etc.)

        Returns:
            Dict contenant la réponse LLM et les résultats intermédiaires.
        """
        # Configuration avec le thread_id pour la persistance
        config = {"configurable": {"thread_id": self.thread_id}}

        # Invoquer le graphe
        result = self.app.invoke(
            {
                "user_query": user_query,
                "language": language,
                "messages": [],
                "selected_files": [],
                "similar_chunks": [],
                "context_text": "",
                "response": "",
            },
            config=config,
        )

        return {
            "selected_files": result.get("selected_files", []),
            "similar_chunks": result.get("similar_chunks", []),
            "response": result.get("response", ""),
        }

    def get_history(self) -> List[Dict[str, str]]:
        """Récupère l'historique de la conversation."""
        config = {"configurable": {"thread_id": self.thread_id}}
        state = self.app.get_state(config)

        history = []
        if state and state.values:
            for msg in state.values.get("messages", []):
                if isinstance(msg, HumanMessage):
                    history.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    history.append({"role": "assistant", "content": msg.content})

        return history


def run_workflow(
    user_query: str,
    csv_path: str = "data/files_index.csv",
    top_k: int = 5,
) -> Dict[str, object]:
    """
    Exécute le workflow complet (mode single-shot sans mémoire).

    Args:
        user_query: Question de l'utilisateur.
        csv_path: Chemin du fichier CSV d'index des documents.
        top_k: Nombre de chunks similaires à récupérer.

    Returns:
        Dict contenant la réponse LLM et les résultats intermédiaires.
    """
    conv = ConversationWorkflow(csv_path=csv_path, top_k=top_k)
    return conv.chat(user_query)


if __name__ == "__main__":
    # Mode conversation avec mémoire via LangGraph MemorySaver
    conv = ConversationWorkflow(top_k=10)

    print("=== MODE CONVERSATION (LangGraph MemorySaver) ===")
    print("Commandes disponibles:")
    print("  - 'quit' : quitter")
    print("  - 'clear' : effacer l'historique")
    print("  - 'history' : afficher l'historique\n")

    while True:
        query = input("Vous: ").strip()

        if query.lower() == "quit":
            print("Au revoir!")
            break
        elif query.lower() == "clear":
            conv.clear_history()
            print("Historique effacé (nouveau thread créé).\n")
            continue
        elif query.lower() == "history":
            history = conv.get_history()
            if history:
                print("\n=== HISTORIQUE ===")
                for i, msg in enumerate(history, 1):
                    role = "Vous" if msg["role"] == "user" else "Assistant"
                    print(f"{i}. [{role}]: {msg['content'][:100]}...")
                print()
            else:
                print("Aucun historique.\n")
            continue
        elif not query:
            continue

        result = conv.chat(query)
        print(f"\nAssistant: {result['response']}\n")
