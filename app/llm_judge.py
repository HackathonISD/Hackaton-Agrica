import os
import re
import pandas as pd
from openai import OpenAI, OpenAIError
from dotenv import load_dotenv
from typing import Any, Optional, List, Dict
from datetime import datetime
from pathlib import Path


class LLMJudge:
    """LLM-as-a-Judge for evaluating document relevance."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """Initialize the LLM Judge client."""
        load_dotenv()
        self.api_key = api_key or os.getenv("OPENAI_TOKEN", "")
        self.base_url = base_url or os.getenv("PROVIDER_URL", "")
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.model = "openai-gpt-5.2"
        self.markdown_base_path = Path("data/markdown_output/Corpus_Offres-Produits_AGRICA")

    def generate_response(
        self,
        user_prompt: str,
        system_prompt: str,
        temperature: float = 0.2,
    ) -> str:
        """Generate a response from the LLM."""
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
            return "No content in response."
        except OpenAIError as e:
            print(f"Error: {e}")
            return "ERROR"

    def find_markdown_file(self, pdf_name: str) -> Optional[Path]:
        """
        Find the markdown file corresponding to a PDF name.
        
        Args:
            pdf_name: Name of the PDF file (e.g., '82468_ccpma_santé_oda_cg.pdf')
        
        Returns:
            Path to the markdown file or None if not found.
        """
        # Remove .pdf extension and add .md
        md_name = pdf_name.replace(".pdf", ".md")
        
        # Search in all subdirectories
        for subdir in self.markdown_base_path.iterdir():
            if subdir.is_dir():
                md_path = subdir / md_name
                if md_path.exists():
                    return md_path
        
        return None

    def read_markdown_content(self, md_path: Path, max_chars: int = 100000) -> str:
        """
        Read markdown file content (truncated if too long).
        
        Args:
            md_path: Path to the markdown file.
            max_chars: Maximum characters to read.
        
        Returns:
            Content of the markdown file.
        """
        try:
            with open(md_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            if len(content) > max_chars:
                return content[:max_chars] + "\n\n[... contenu tronqué ...]"
            return content
        except Exception as e:
            return f"Erreur de lecture: {e}"

    def evaluate_document_relevance(
        self,
        question: str,
        document_name: str,
        document_content: str,
    ) -> Dict[str, Any]:
        """
        Evaluate if a document is relevant to a question.
        
        Args:
            question: The user's question.
            document_name: Name of the document.
            document_content: Content of the document.
        
        Returns:
            Dict with score and justification.
        """
        system_prompt = """Tu es un juge expert chargé d'évaluer la pertinence d'un document pour répondre à une question.

Tu dois évaluer si le document contient des informations utiles pour répondre à la question posée.

IMPORTANT - FORMAT DE RÉPONSE OBLIGATOIRE:
Tu dois répondre EXACTEMENT dans ce format (3 lignes):
SCORE: [0-10]
PERTINENT: [OUI/NON]
JUSTIFICATION: [Explication en 1-2 phrases]

Critères de notation:
- 0-2: Document non pertinent, aucune information utile
- 3-4: Document faiblement pertinent, informations indirectes
- 5-6: Document moyennement pertinent, quelques informations utiles
- 7-8: Document pertinent, contient des informations directement utiles
- 9-10: Document très pertinent, répond directement à la question"""

        user_prompt = f"""Question de l'utilisateur: {question}

Document à évaluer: {document_name}

Contenu du document:
{document_content}

Évalue la pertinence de ce document pour répondre à la question."""

        response = self.generate_response(user_prompt, system_prompt, temperature=0.1)
        
        # Parse response
        result = {
            "document": document_name,
            "score": 0,
            "pertinent": "INCONNU",
            "justification": response,
        }
        
        try:
            lines = response.strip().split("\n")
            for line in lines:
                if line.startswith("SCORE:"):
                    score_match = re.search(r'\d+', line)
                    if score_match:
                        result["score"] = int(score_match.group())
                elif line.startswith("PERTINENT:"):
                    result["pertinent"] = "OUI" if "OUI" in line.upper() else "NON"
                elif line.startswith("JUSTIFICATION:"):
                    result["justification"] = line.replace("JUSTIFICATION:", "").strip()
        except Exception:
            pass
        
        return result

    def evaluate_all_results(
        self,
        results_csv_path: str,
        output_path: str,
        max_docs_per_question: int = 3,
    ) -> pd.DataFrame:
        """
        Evaluate all documents from a results CSV file.
        
        Args:
            results_csv_path: Path to the CSV file with questions and documents.
            output_path: Path to save the evaluation results.
            max_docs_per_question: Max documents to evaluate per question (to save API calls).
        
        Returns:
            DataFrame with evaluation results.
        """
        # Read results CSV
        df = pd.read_csv(results_csv_path)
        
        all_evaluations = []
        
        print(f"📋 Évaluation de {len(df)} questions...\n")
        
        for idx, row in df.iterrows():
            question = row["question"]
            fichiers = row["fichiers_pertinents"]
            
            print(f"\n[{idx + 1}/{len(df)}] Question: {question[:50]}...")
            
            if fichiers == "AUCUN" or pd.isna(fichiers):
                all_evaluations.append({
                    "question": question,
                    "document": "AUCUN",
                    "score": None,
                    "pertinent": None,
                    "justification": "Aucun document sélectionné",
                    "fichier_trouve": False,
                })
                continue
            
            # Split documents
            docs = [d.strip() for d in fichiers.split("|")][:max_docs_per_question]
            
            for doc_name in docs:
                print(f"   → Évaluation: {doc_name[:40]}...")
                
                # Find and read markdown file
                md_path = self.find_markdown_file(doc_name)
                
                if md_path is None:
                    all_evaluations.append({
                        "question": question,
                        "document": doc_name,
                        "score": None,
                        "pertinent": None,
                        "justification": "Fichier markdown non trouvé",
                        "fichier_trouve": False,
                    })
                    continue
                
                # Read content
                content = self.read_markdown_content(md_path)
                
                # Evaluate relevance
                evaluation = self.evaluate_document_relevance(question, doc_name, content)
                
                all_evaluations.append({
                    "question": question,
                    "document": doc_name,
                    "score": evaluation["score"],
                    "pertinent": evaluation["pertinent"],
                    "justification": evaluation["justification"],
                    "fichier_trouve": True,
                })
                
                print(f"      Score: {evaluation['score']}/10 - {evaluation['pertinent']}")
        
        # Create results DataFrame
        eval_df = pd.DataFrame(all_evaluations)
        
        # Save to CSV
        eval_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"\n✅ Évaluations sauvegardées dans: {output_path}")
        
        return eval_df

    def print_summary(self, eval_df: pd.DataFrame):
        """Print a summary of the evaluation results."""
        print("\n" + "=" * 60)
        print("📊 RÉSUMÉ DE L'ÉVALUATION")
        print("=" * 60)
        
        # Filter only evaluated documents
        evaluated = eval_df[eval_df["score"].notna()]
        
        if len(evaluated) == 0:
            print("Aucun document évalué.")
            return
        
        print(f"\n📁 Documents évalués: {len(evaluated)}")
        print(f"📄 Fichiers trouvés: {len(eval_df[eval_df['fichier_trouve'] == True])}")
        print(f"❌ Fichiers non trouvés: {len(eval_df[eval_df['fichier_trouve'] == False])}")
        
        avg_score = evaluated["score"].mean()
        print(f"\n📈 Score moyen: {avg_score:.2f}/10")
        
        pertinent_count = len(evaluated[evaluated["pertinent"] == "OUI"])
        non_pertinent_count = len(evaluated[evaluated["pertinent"] == "NON"])
        
        print(f"✅ Documents pertinents: {pertinent_count} ({100*pertinent_count/len(evaluated):.1f}%)")
        print(f"❌ Documents non pertinents: {non_pertinent_count} ({100*non_pertinent_count/len(evaluated):.1f}%)")
        
        # Score distribution
        print("\n📊 Distribution des scores:")
        for score_range, label in [(range(0, 3), "0-2 (Non pertinent)"), 
                                    (range(3, 5), "3-4 (Faible)"),
                                    (range(5, 7), "5-6 (Moyen)"),
                                    (range(7, 9), "7-8 (Pertinent)"),
                                    (range(9, 11), "9-10 (Très pertinent)")]:
            count = len(evaluated[evaluated["score"].isin(score_range)])
            bar = "█" * (count // 2) if count > 0 else ""
            print(f"   {label}: {count} {bar}")


if __name__ == "__main__":
    judge = LLMJudge()
    
    # Path to the results file
    results_csv = "exports/resultats_20260131_033120.csv"
    output_path = f"exports/evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    # Check if file exists
    if not os.path.exists(results_csv):
        print(f"❌ Fichier non trouvé: {results_csv}")
        print("Recherche du fichier le plus récent...")
        
        exports_dir = Path("exports")
        csv_files = list(exports_dir.glob("resultats_*.csv"))
        if csv_files:
            results_csv = str(max(csv_files, key=os.path.getmtime))
            print(f"✅ Fichier trouvé: {results_csv}")
        else:
            print("❌ Aucun fichier de résultats trouvé.")
            exit(1)
    
    print(f"📂 Lecture de: {results_csv}")
    print(f"📁 Dossier markdown: {judge.markdown_base_path}")
    
    # Evaluate (max 3 docs per question to save API calls)
    eval_df = judge.evaluate_all_results(
        results_csv, 
        output_path,
        max_docs_per_question=3  # Limite pour économiser les appels API
    )
    
    # Print summary
    judge.print_summary(eval_df)
