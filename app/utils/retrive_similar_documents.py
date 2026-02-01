"""
Module pour récupérer les documents similaires depuis la base de données vectorielle FAISS.
"""

import os
import pickle
from typing import List, Dict, Tuple

import faiss
import numpy as np
from embed_document import embed_document


def load_vector_database(database_folder: str) -> Tuple[faiss.IndexFlatL2, List[Dict]]:
    """
    Charge l'index FAISS et les métadonnées depuis le dossier spécifié.

    Args:
        database_folder: Chemin vers le dossier contenant l'index et les métadonnées

    Returns:
        Tuple[faiss.IndexFlatL2, List[Dict]]: L'index FAISS et la liste des métadonnées
    """
    index_path = os.path.join(database_folder, "faiss_index.bin")
    metadata_path = os.path.join(database_folder, "metadata.pkl")

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index FAISS introuvable: {index_path}")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Métadonnées introuvables: {metadata_path}")

    # Charger l'index FAISS
    index = faiss.read_index(index_path)

    # Charger les métadonnées
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    return index, metadata


def retrieve_similar_documents(
    query: str,
    database_folder: str = None,
    model_name: str = "intfloat/multilingual-e5-base",
    top_k: int = 5,
    file_names: List[str] = None,
) -> List[Dict]:
    """
    Recherche les documents les plus similaires à la requête dans la base de données vectorielle.

    Args:
        query: La requête de recherche (texte)
        database_folder: Chemin vers le dossier contenant l'index FAISS (optionnel)
        model_name: Nom du modèle d'embedding (doit être le même que lors de la création)
        top_k: Nombre de documents à retourner
        file_names: Liste des noms de fichiers à filtrer (optionnel). Si spécifié, seuls les chunks
                   provenant de ces fichiers seront retournés.

    Returns:
        List[Dict]: Liste des documents similaires avec leurs métadonnées et scores
        Chaque dict contient: {"source", "text", "score", "rank"}

    Example:
        results = retrieve_similar_documents("prévoyance santé", top_k=5)
        results = retrieve_similar_documents("prévoyance santé", top_k=5, file_names=["document1.md", "document2.md"])
        for result in results:
            print(f"Score: {result['score']}, Source: {result['source']}")
    """
    # Définir le dossier par défaut
    if database_folder is None:
        database_folder = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "vector_database",
        )

    # Charger l'index et les métadonnées
    print(f"Chargement de la base de données depuis: {database_folder}")
    index, metadata = load_vector_database(database_folder)
    print(f"✓ Index chargé: {index.ntotal} vecteurs")

    # Créer l'embedding de la requête
    print(f"Création de l'embedding pour la requête: '{query}'")
    query_embedding = embed_document(query, model_name)
    query_vector = np.array([query_embedding]).astype("float32")

    # Rechercher plus de candidats si on filtre par fichiers
    search_k = top_k * 10 if file_names else top_k
    distances, indices = index.search(query_vector, search_k)

    # Préparer les résultats
    results = []
    for distance, idx in zip(distances[0], indices[0]):
        if idx >= len(metadata):
            continue

        source = metadata[idx]["source"]

        # Filtrer par noms de fichiers si spécifié
        if file_names:
            source_filename = os.path.basename(source)
            # Enlever l'extension pour comparer (PDF vs MD)
            source_name_no_ext = os.path.splitext(source_filename)[0]

            # Vérifier si un des noms de fichiers correspond
            matched = False
            for fn in file_names:
                fn_no_ext = os.path.splitext(fn)[0]
                # Comparer sans extension ou avec nom partiel
                if (
                    fn_no_ext in source_name_no_ext
                    or source_name_no_ext in fn_no_ext
                    or fn in source_filename
                    or source_filename in fn
                ):
                    matched = True
                    break

            if not matched:
                continue

        result = {
            "rank": len(results) + 1,
            "score": float(distance),
            "source": source,
            "text": metadata[idx]["text"],
        }
        results.append(result)

        if len(results) >= top_k:
            break

    print(f"✓ {len(results)} documents similaires trouvés")
    if file_names:
        print(f"  Filtrés par fichiers: {', '.join(file_names)}")
    return results


if __name__ == "__main__":
    """Exemple d'utilisation."""
    # Exemple de recherche
    query = "En cas d’arrêt de travail d’origine professionnelle quelle est la Prestation versée"
    results = retrieve_similar_documents(
        query,
        database_folder=os.path.join(os.getcwd(), "data/vector_database"),
        file_names=["202201_TG_CCN52 Av51_Ref131.pdf"],
    )

    for result in results:
        print(f"[{result['rank']}] Score: {result['score']:.4f}")
        print(f"Source: {result['source']}")
        text_preview = result["text"]
        print(f"Texte: {text_preview}")
        print("-" * 80)
