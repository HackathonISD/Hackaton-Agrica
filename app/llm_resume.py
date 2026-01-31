import os
import pandas as pd
from openai import OpenAI, OpenAIError
from dotenv import load_dotenv
from typing import Any, Optional, List, Dict


class LLmClient:
    """A client class for interacting with OpenAI API."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize the OpenAI client.

        Args:
            api_key: OpenAI API key. If None, reads from OPENAI_TOKEN env variable.
            base_url: Base URL for the API. If None, reads from PROVIDER_URL env variable.
        """
        load_dotenv()
        self.api_key = api_key or os.getenv("OPENAI_TOKEN", "")
        self.base_url = base_url or os.getenv("PROVIDER_URL", "")
        self.client = self._initialize_client()
        self.model = "openai-gpt-5.2"

    def _initialize_client(self) -> OpenAI:
        """Initialize and return the OpenAI client."""
        return OpenAI(api_key=self.api_key, base_url=self.base_url)

    def generate_prompt(self, prompt_template: str, **kwargs: Any) -> str:
        """
        Generate a formatted prompt from a template.

        Args:
            prompt_template: Template string with placeholders.
            **kwargs: Values to fill the template placeholders.

        Returns:
            Formatted prompt string.
        """
        return prompt_template.format(**kwargs)

    def generate_response(
        self,
        user_prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        temperature: float = 0.5,
    ) -> str:
        """
        Generate a response from OpenAI.

        Args:
            user_prompt: The user's input prompt.
            system_prompt: The system prompt to set context.
            temperature: Controls randomness (0-2). Default is 0.5.
            max_tokens: Maximum tokens in the response.

        Returns:
            The generated response content.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
            )

            if response.choices[0].message.content is not None:
                return response.choices[0].message.content
            else:
                return "No content in response."

        except OpenAIError as e:
            print(f"An error occurred: {e}")
            return "An error occurred while generating the response."

    def load_document_index(self, csv_path: str) -> pd.DataFrame:
        """
        Load the document index from a CSV file.

        Args:
            csv_path: Path to the CSV file containing document metadata.

        Returns:
            DataFrame with document information.
        """
        df = pd.read_csv(csv_path)
        # Remove empty rows
        df = df.dropna(subset=["nom_pdf"])
        return df

    def select_relevant_documents(
        self,
        user_question: str,
        documents_df: pd.DataFrame,
    ) -> List[Dict[str, str]]:
        """
        Select relevant documents based on user question using LLM.

        Args:
            user_question: The user's question.
            documents_df: DataFrame with columns 'nom_pdf', 'résumé', 'tags'.

        Returns:
            List of relevant documents with their metadata.
        """
        # Build document list for the prompt
        doc_list = []
        for idx, row in documents_df.iterrows():
            doc_list.append(
                f"[{idx}] Fichier: {row['nom_pdf']}\n"
                f"    Résumé: {row['résumé']}"
            )

        documents_text = "\n\n".join(doc_list)

        system_prompt = """Tu es un assistant expert en recherche documentaire pour AGRICA (protection sociale agricole).
Ta tâche est de sélectionner les documents les plus pertinents pour répondre à la question de l'utilisateur.

Analyse la question et compare-la avec les tags et résumés des documents disponibles.
Retourne UNIQUEMENT les indices des documents pertinents, séparés par des virgules.
Par exemple: 0, 3, 5

Si aucun document n'est pertinent, retourne: AUCUN

Sois précis et ne sélectionne que les documents vraiment utiles pour répondre à la question."""

        user_prompt = f"""Question de l'utilisateur: {user_question}

Documents disponibles:
{documents_text}

Quels sont les indices des documents pertinents pour répondre à cette question? """

        response = self.generate_response(user_prompt, system_prompt, temperature=0.2)

        # Parse response to get document indices
        selected_docs = []
        if "AUCUN" not in response.upper():
            try:
                indices = [int(idx.strip()) for idx in response.split(",") if idx.strip().isdigit()]
                for idx in indices:
                    if idx in documents_df.index:
                        row = documents_df.loc[idx]
                        selected_docs.append({
                            "fichier": row["nom_pdf"],
                            "résumé": row["résumé"],
                            "tags": row["tags"],
                        })
            except ValueError:
                pass

        return selected_docs

    def answer_with_documents(
        self,
        user_question: str,
        documents_df: pd.DataFrame,
    ) -> str:
        """
        Answer user question by first selecting relevant documents.

        Args:
            user_question: The user's question.
            documents_df: DataFrame with document metadata.

        Returns:
            Response with selected documents and answer.
        """
        # Select relevant documents
        relevant_docs = self.select_relevant_documents(user_question, documents_df)

        if not relevant_docs:
            return "Aucun document pertinent trouvé pour votre question."

        # Format selected documents
        docs_info = "\n".join([
            f"📄 **{doc['fichier']}**\n   resumé: {doc['résumé'][:100]}..."
            for doc in relevant_docs
        ])

        return f"Documents pertinents sélectionnés:\n\n{docs_info}"


if __name__ == "__main__":
    llm_client = LLmClient()

    # Load document index
    csv_path = "data/RAG - Sheet1.csv"
    try:
        documents_df = llm_client.load_document_index(csv_path)
        print(f"✅ RESULMé only:  {len(documents_df)} documents chargés depuis {csv_path}\n")
    except FileNotFoundError:
        print(f"❌ Fichier CSV non trouvé: {csv_path}")
        exit(1)

    # Example question about health insurance
    user_question = "quelle est ma garantie décès ?"

    print(f"Question: {user_question}\n")
    print("-" * 50)

    # Get relevant documents
    result = llm_client.answer_with_documents(user_question, documents_df)
    print(result)