import os
import re
import pandas as pd
from openai import OpenAI, OpenAIError
from dotenv import load_dotenv
from typing import Any, Optional, List, Dict
from datetime import datetime


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
            max_documents: Maximum number of documents to return.

        Returns:
            List of relevant documents with their metadata.
        """
        # Build document list for the prompt
        doc_list = []
        for idx, row in documents_df.iterrows():
            doc_list.append(
                f"[{idx}] Fichier: {row['nom_pdf']}\n"
                f"    Tags: {row['tags']}\n"
                f"    Résumé: {row['résumé']}"
            )

        documents_text = "\n\n".join(doc_list)

        system_prompt = """Tu es un assistant expert en recherche documentaire pour AGRICA (protection sociale agricole).
Ta tâche est de sélectionner les documents les plus pertinents pour répondre à la question de l'utilisateur.

Analyse la question et compare-la avec les tags et résumés des documents disponibles.

IMPORTANT - FORMAT DE RÉPONSE OBLIGATOIRE:
- Retourne UNIQUEMENT les numéros des documents pertinents séparés par des virgules
- Exemple de réponse correcte: 0, 3, 5
- Ne mets PAS de texte explicatif, JUSTE les numéros
- Si aucun document n'est pertinent, retourne exactement: AUCUN

Sois précis et ne sélectionne que les documents vraiment utiles."""

        user_prompt = f"""Question de l'utilisateur: {user_question}

Documents disponibles:
{documents_text}

Réponds UNIQUEMENT avec les numéros des documents pertinents (ex: 0, 3, 5) ou AUCUN:"""

        response = self.generate_response(user_prompt, system_prompt, temperature=0.1)

        # Parse response to get document indices - robust parsing with regex
        selected_docs = []
        if "AUCUN" not in response.upper():
            try:
                # Extract all numbers from the response using regex
                indices = [int(num) for num in re.findall(r'\d+', response)]
                # Remove duplicates while preserving order
                seen = set()
                unique_indices = [x for x in indices if not (x in seen or seen.add(x))]
                
                for idx in unique_indices:
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
            f"📄 **{doc['fichier']}**\n   Tags: {doc['tags'][:100]}..."
            for doc in relevant_docs
        ])

        return f"Documents pertinents sélectionnés:\n\n{docs_info}"

    def load_questions(self, questions_path: str) -> List[str]:
        """
        Load questions from a text file.

        Args:
            questions_path: Path to the questions file.

        Returns:
            List of questions.
        """
        with open(questions_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        # Skip header if present and filter empty lines
        questions = [line.strip() for line in lines[1:] if line.strip()]
        return questions

    def process_questions_and_save(
        self,
        questions_path: str,
        documents_df: pd.DataFrame,
        output_path: str,
    ) -> pd.DataFrame:
        """
        Process all questions and save results to CSV.

        Args:
            questions_path: Path to the questions file.
            documents_df: DataFrame with document metadata.
            output_path: Path to save the results CSV.

        Returns:
            DataFrame with results.
        """
        questions = self.load_questions(questions_path)
        results = []

        print(f"📋 Traitement de {len(questions)} questions...\n")

        for i, question in enumerate(questions, 1):
            print(f"[{i}/{len(questions)}] {question[:60]}...")
            
            relevant_docs = self.select_relevant_documents(question, documents_df)
            
            if relevant_docs:
                fichiers = " | ".join([doc["fichier"] for doc in relevant_docs])
                nb_docs = len(relevant_docs)
            else:
                fichiers = "AUCUN"
                nb_docs = 0
            
            results.append({
                "question": question,
                "nb_documents": nb_docs,
                "fichiers_pertinents": fichiers,
            })
            
            print(f"   → {nb_docs} document(s) trouvé(s)\n")

        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Save to CSV
        results_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"✅ Résultats sauvegardés dans: {output_path}")
        
        return results_df


if __name__ == "__main__":
    llm_client = LLmClient()

    # Paths
    csv_path = "data/RAG - Sheet1.csv"
    questions_path = "data/question.txt"
    output_path = f"exports/resultats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    # Load document index
    try:
        documents_df = llm_client.load_document_index(csv_path)
        print(f"✅ {len(documents_df)} documents chargés depuis {csv_path}\n")
    except FileNotFoundError:
        print(f"❌ Fichier CSV non trouvé: {csv_path}")
        exit(1)

    # Process all questions and save results
    try:
        results_df = llm_client.process_questions_and_save(
            questions_path, documents_df, output_path
        )
        
        print("\n" + "=" * 50)
        print("📊 Résumé:")
        print(f"   - Questions traitées: {len(results_df)}")
        print(f"   - Questions avec résultats: {len(results_df[results_df['nb_documents'] > 0])}")
        print(f"   - Questions sans résultats: {len(results_df[results_df['nb_documents'] == 0])}")
        
    except FileNotFoundError:
        print(f"❌ Fichier de questions non trouvé: {questions_path}")
        exit(1)
    
    