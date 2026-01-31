"""
Module pour créer une base de données vectorielle FAISS avec les embeddings des documents markdown.
Les tableaux markdown sont convertis en phrases textuelles et les chunks sont tronqués
au maximum de tokens supporté par le modèle d'embedding.
"""

import os
import pickle
import re
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from Hackaton.app.utils.embed_document import embed_document


# =========================================================
# Chargement des fichiers markdown
# =========================================================


def load_markdown_files(markdown_folder: str) -> List[Tuple[str, str]]:
    markdown_files = []
    markdown_path = Path(markdown_folder)

    if not markdown_path.exists():
        raise FileNotFoundError(f"Le dossier {markdown_folder} n'existe pas")

    for md_file in markdown_path.rglob("*.md"):
        try:
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()
                markdown_files.append((str(md_file), content))
                print(f"✓ Chargé: {md_file.name}")
        except Exception as e:
            print(f"✗ Erreur lors du chargement de {md_file.name}: {str(e)}")

    print(f"\nTotal: {len(markdown_files)} fichiers markdown chargés")
    return markdown_files


# =========================================================
# Conversion des tableaux markdown en phrases
# =========================================================


def markdown_table_to_sentences(table_lines: List[str]) -> List[str]:
    rows = [
        [cell.strip() for cell in line.strip("|").split("|")]
        for line in table_lines
        if line.strip() and not all(ch in "-: " for ch in line)
    ]

    if len(rows) < 2:
        return []

    headers = rows[0]
    data_rows = rows[1:]
    sentences = []

    for row in data_rows:
        if len(row) != len(headers):
            continue

        parts = []
        for header, value in zip(headers, row):
            if value:
                parts.append(f"{header} : {value}")

        if parts:
            sentences.append(" ; ".join(parts) + ".")

    return sentences


# =========================================================
# Nettoyage + transformation markdown
# =========================================================


def preprocess_markdown(text: str) -> str:
    # Remove HTML comments
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)

    # Remove long lines of dashes (pure separators)
    text = re.sub(r"^-+$", "", text, flags=re.MULTILINE)

    lines = text.splitlines()
    cleaned_lines = []
    table_buffer = []
    inside_table = False

    for line in lines:
        stripped = line.strip()
        is_table_line = stripped.startswith("|") and "|" in stripped

        if is_table_line:
            inside_table = True
            table_buffer.append(line)
            continue

        if inside_table and not is_table_line:
            cleaned_lines.extend(markdown_table_to_sentences(table_buffer))
            table_buffer = []
            inside_table = False

        if stripped and not re.fullmatch(r"-{5,}", stripped):
            cleaned_lines.append(stripped)

    if table_buffer:
        cleaned_lines.extend(markdown_table_to_sentences(table_buffer))

    cleaned_text = "\n".join(cleaned_lines)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text).strip()

    return cleaned_text


# =========================================================
# Validation des chunks (compatible données numériques)
# =========================================================


def is_valid_chunk(text: str, min_length: int = 100) -> bool:
    if len(text.strip()) < min_length:
        return False

    alpha_count = sum(c.isalpha() for c in text)
    digit_count = sum(c.isdigit() for c in text)

    if alpha_count < 30 and digit_count < 10:
        return False

    lines = text.split("\n")
    empty_lines = sum(1 for line in lines if not line.strip())
    if len(lines) > 0 and (empty_lines / len(lines)) > 0.7:
        return False

    return True


# =========================================================
# Découpage en chunks
# =========================================================


def split_documents(
    documents: List[Tuple[str, str]],
    tokenizer_name: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> List[Tuple[str, str]]:

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n## ",
            "\n### ",
            "\n#### ",
            "\n\n",
            "\n",
            ". ",
            " ",
        ],
    )

    chunks = []
    filtered_count = 0

    for file_path, content in documents:
        content = preprocess_markdown(content)
        split_texts = splitter.split_text(content)

        for text in split_texts:
            if is_valid_chunk(text):
                chunks.append((file_path, text))
            else:
                filtered_count += 1

    print(f"\nTotal: {len(chunks)} chunks créés ({filtered_count} chunks filtrés)")
    return chunks


# =========================================================
# Troncature stricte au max tokens du modèle
# =========================================================


def truncate_to_max_tokens(text: str, tokenizer, max_tokens: int = 512) -> str:
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_tokens,
        return_tensors=None,
    )
    return tokenizer.decode(encoded["input_ids"], skip_special_tokens=True)


# =========================================================
# Création de la base vectorielle FAISS
# =========================================================


def create_vector_database(
    markdown_folder: str,
    output_folder: str = None,
    model_name: str = "intfloat/multilingual-e5-base",
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> Tuple[faiss.IndexFlatL2, List[dict]]:

    if output_folder is None:
        output_folder = os.path.join(os.getcwd(), "data/vector_database")

    os.makedirs(output_folder, exist_ok=True)

    print("=" * 60)
    print("CRÉATION DE LA BASE DE DONNÉES VECTORIELLE")
    print("=" * 60)

    print("\n[1/4] Chargement des fichiers markdown...")
    documents = load_markdown_files(markdown_folder)

    if not documents:
        raise ValueError("Aucun fichier markdown trouvé")

    print("\n[2/4] Découpage en chunks...")
    chunks = split_documents(documents, model_name, chunk_size, chunk_overlap)

    print("\n[3/4] Création des embeddings...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    texts = [
        truncate_to_max_tokens(text, tokenizer, max_tokens=512) for _, text in chunks
    ]

    embeddings = embed_document(texts, model_name)
    embeddings_array = np.array(embeddings).astype("float32")

    max_len = max(len(tokenizer(t)["input_ids"]) for t in texts)
    print(f"✓ Longueur max après troncature: {max_len} tokens")
    print(f"✓ Embeddings créés - Shape: {embeddings_array.shape}")

    print("\n[4/4] Création de l'index FAISS...")
    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_array)

    index_path = os.path.join(output_folder, "faiss_index.bin")
    faiss.write_index(index, index_path)

    metadata = [{"source": src, "text": txt} for src, txt in chunks]
    metadata_path = os.path.join(output_folder, "metadata.pkl")
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)

    print(f"✓ Index sauvegardé: {index_path}")
    print(f"✓ Métadonnées sauvegardées: {metadata_path}")
    print("=" * 60)

    return index, metadata


if __name__ == "__main__":
    markdown_folder = os.path.join(os.getcwd(), "data/markdown_output")

    if not os.path.exists(markdown_folder):
        print(f"Erreur: Le dossier {markdown_folder} n'existe pas")

    create_vector_database(markdown_folder)
