import re
from typing import Dict, List

import pandas as pd

from llm import LLmClient


class RelevantDocumentsSelector:
    def __init__(
        self,
        user_question: str,
        llm_client: LLmClient | None = None,
    ) -> None:
        self.user_question = user_question
        self.llm_client = llm_client or LLmClient()

    def select_relevant_documents(self, csv_path: str) -> List[Dict[str, str]]:
        """
        Select relevant documents based on user question using LLM.

        Args:
            csv_path: Path to the CSV file containing document metadata.

        Returns:
            List of relevant documents with their metadata.
        """
        documents_df = pd.read_csv(csv_path)
        documents_df = documents_df.dropna(subset=["nom_pdf"])

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

        user_prompt = f"""Question de l'utilisateur: {self.user_question}

Documents disponibles:
{documents_text}

Réponds UNIQUEMENT avec les numéros des documents pertinents (ex: 0, 3, 5) ou AUCUN:"""

        response = self.llm_client.generate_response(
            user_prompt, system_prompt, temperature=0.1
        )

        selected_docs: List[Dict[str, str]] = []
        if "AUCUN" not in response.upper():
            indices = [int(num) for num in re.findall(r"\d+", response)]
            seen = set()
            unique_indices = [x for x in indices if not (x in seen or seen.add(x))]

            for idx in unique_indices:
                if idx in documents_df.index:
                    row = documents_df.loc[idx]
                    selected_docs.append(
                        {
                            "fichier": row["nom_pdf"],
                            "résumé": row["résumé"],
                            "tags": row["tags"],
                        }
                    )

        return selected_docs


if __name__ == "__main__":
    csv_path = "data/files_index.csv"
    question = "Quelles sont les prestations en cas d'arrêt de travail d'origine professionnelle ?"
    selector = RelevantDocumentsSelector(question)

    selected_relevant_docs = selector.select_relevant_documents(csv_path)

    print("Documents pertinents sélectionnés:")
    for doc in selected_relevant_docs:
        print(f"- Fichier: {doc['fichier']}")
        print(f"  Résumé: {doc['résumé']}")
        print(f"  Tags: {doc['tags']}\n")
