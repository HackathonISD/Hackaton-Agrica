"""
Module pour créer les embeddings des documents à l'aide de modèles de transformers.
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Union, List


def embed_document(
    document: str, model_name: str = "intfloat/multilingual-e5-base"
) -> List[float]:
    """
    Charge un modèle et génère l'embedding pour un document unique.

    Args:
        document: Le document à encoder
        model_name: Nom du modèle à charger (par défaut: dangvantuan/sentence-camembert-base)

    Returns:
        List[float]: L'embedding du document (vecteur)

    Example:
        embedding = embed_single_document("Bonjour", "dangvantuan/sentence-camembert-base")
    """

    model = SentenceTransformer(model_name)

    embeddings = model.encode(document)

    # normalize the embeddings
    embeddings = embeddings / np.linalg.norm(embeddings)

    return embeddings.tolist()


if __name__ == "__main__":
    # Exemple d'utilisation
    doc = "Ceci est un exemple de document."
    embedding = embed_document(doc)
    norm = np.linalg.norm(embedding)
    print(f"Norm of the embedding vector: {norm}")
