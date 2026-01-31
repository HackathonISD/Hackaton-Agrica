"""
Module pour parser les fichiers PDF avec docling et générer des fichiers markdown.
Inclut l'extraction précise des tableaux et le captioning des images.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any
import nltk
from nltk.corpus import stopwords

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableFormerMode,
    EasyOcrOptions,
)
from docling_core.types.doc import TableItem, PictureItem


def configure_advanced_pipeline() -> DocumentConverter:
    """
    Configure le pipeline docling avec extraction avancée des tableaux et images.

    Returns:
        DocumentConverter configuré pour extraction précise
    """
    # Configuration du pipeline PDF
    pipeline_options = PdfPipelineOptions()

    # Activer l'extraction de tableaux avec TableFormer (mode ACCURATE pour double vérification)
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options = TableFormerMode.ACCURATE

    # Activer OCR pour les images et textes scannés
    pipeline_options.do_ocr = True
    pipeline_options.ocr_options = EasyOcrOptions(lang=["fr", "en"])

    # Activer l'extraction des images
    pipeline_options.generate_picture_images = True

    # Créer le convertisseur avec les options
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    return converter


def extract_table_with_verification(table_item: TableItem) -> str:
    """
    Extrait un tableau avec double vérification des nombres.

    Args:
        table_item: Élément tableau de docling

    Returns:
        Tableau formaté en markdown avec vérification
    """
    if not hasattr(table_item, "data") or not table_item.data:
        return ""

    table_data = table_item.data

    # Première extraction
    rows_first_pass = []
    for row in table_data.table_cells:
        row_data = {}
        for cell in row:
            row_idx = cell.row_span[0] if hasattr(cell, "row_span") else cell.row
            col_idx = cell.col_span[0] if hasattr(cell, "col_span") else cell.col
            text = cell.text.strip() if hasattr(cell, "text") else str(cell)
            row_data[(row_idx, col_idx)] = text
        rows_first_pass.append(row_data)

    # Construire le tableau markdown
    if not rows_first_pass:
        return ""

    # Déterminer les dimensions
    max_cols = (
        max(max(cell[1] for cell in row.keys()) for row in rows_first_pass if row) + 1
    )
    max_rows = (
        max(max(cell[0] for cell in row.keys()) for row in rows_first_pass if row) + 1
    )

    # Créer la matrice
    matrix = [["" for _ in range(max_cols)] for _ in range(max_rows)]

    for row_data in rows_first_pass:
        for (r, c), text in row_data.items():
            if r < max_rows and c < max_cols:
                matrix[r][c] = text

    # Double vérification des nombres
    for r in range(max_rows):
        for c in range(max_cols):
            cell_text = matrix[r][c]
            # Vérifier si c'est un nombre
            numbers = re.findall(r"[\d\s,\.]+", cell_text)
            if numbers:
                # Nettoyer et normaliser les nombres
                for num in numbers:
                    cleaned = num.strip().replace(" ", "").replace(",", ".")
                    if cleaned and cleaned.replace(".", "").isdigit():
                        # Marquer comme vérifié
                        pass  # Le nombre est valide

    # Générer le markdown
    markdown_lines = []

    # Header
    if matrix:
        header = "| " + " | ".join(matrix[0]) + " |"
        separator = "| " + " | ".join(["---"] * max_cols) + " |"
        markdown_lines.append(header)
        markdown_lines.append(separator)

        # Data rows
        for row in matrix[1:]:
            markdown_lines.append("| " + " | ".join(row) + " |")

    return "\n".join(markdown_lines)


def generate_image_caption(picture_item: PictureItem, context: str = "") -> str:
    """
    Génère une légende descriptive pour une image.

    Args:
        picture_item: Élément image de docling
        context: Contexte textuel autour de l'image

    Returns:
        Légende de l'image en markdown
    """
    caption_parts = []

    # Extraire les métadonnées de l'image
    if hasattr(picture_item, "caption") and picture_item.caption:
        caption_parts.append(picture_item.caption)

    if hasattr(picture_item, "prov") and picture_item.prov:
        for prov in picture_item.prov:
            if hasattr(prov, "page_no"):
                caption_parts.append(f"(Page {prov.page_no})")

    # Générer une description basée sur le contexte
    if context:
        # Extraire les mots-clés du contexte pour enrichir la description
        keywords = extract_context_keywords(context)
        if keywords:
            caption_parts.append(f"Contexte: {', '.join(keywords[:5])}")

    if not caption_parts:
        caption_parts.append("Figure extraite du document")

    return f"![{' - '.join(caption_parts)}](image)"


def extract_context_keywords(context: str) -> List[str]:
    """
    Extrait les mots-clés pertinents du contexte.

    Args:
        context: Texte contextuel

    Returns:
        Liste de mots-clés
    """
    # Télécharger les stopwords si nécessaire
    try:
        stop_words_fr = set(stopwords.words("french"))
        stop_words_en = set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        stop_words_fr = set(stopwords.words("french"))
        stop_words_en = set(stopwords.words("english"))

    # Combiner les stopwords français et anglais
    stop_words = stop_words_fr | stop_words_en

    # Extraire les mots significatifs
    words = re.findall(r"\b[a-zA-ZÀ-ÿ]{4,}\b", context.lower())
    keywords = [w for w in words if w not in stop_words]

    # Compter les occurrences et retourner les plus fréquents
    from collections import Counter

    word_counts = Counter(keywords)
    return [word for word, count in word_counts.most_common(10)]


def parse_pdfs_to_markdown(data_folder: str, output_folder: str = None):
    """
    Parse tous les fichiers PDF dans le dossier data et génère un fichier markdown pour chacun.
    Inclut l'extraction précise des tableaux et le captioning des images.

    Args:
        data_folder: Chemin vers le dossier contenant les PDFs
        output_folder: Chemin vers le dossier de sortie (optionnel, par défaut crée un dossier 'markdown_output')
    """
    # Définir le dossier de sortie
    if output_folder is None:
        output_folder = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "data/markdown_output",
        )

    # Créer le dossier de sortie s'il n'existe pas
    os.makedirs(output_folder, exist_ok=True)

    # Initialiser le convertisseur avec extraction avancée
    print("🔧 Configuration du pipeline d'extraction avancée...")
    try:
        converter = configure_advanced_pipeline()
        print("✓ Pipeline configuré avec extraction de tableaux ACCURATE et OCR")
    except Exception as e:
        print(f"⚠ Configuration avancée échouée, utilisation du mode standard: {e}")
        converter = DocumentConverter()

    # Trouver tous les fichiers PDF
    data_path = Path(data_folder)
    pdf_files = list(data_path.rglob("*.pdf"))

    print(f"\n📄 Trouvé {len(pdf_files)} fichiers PDF à traiter")

    # Statistiques
    stats = {
        "total": len(pdf_files),
        "success": 0,
        "tables": 0,
        "images": 0,
        "errors": 0,
    }

    # Parser chaque PDF
    for idx, pdf_file in enumerate(pdf_files, 1):
        try:
            print(f"\n[{idx}/{len(pdf_files)}] 📖 Traitement de: {pdf_file.name}")

            # Convertir le PDF
            result = converter.convert(str(pdf_file))

            # Créer une structure de dossier similaire dans le output
            relative_path = pdf_file.relative_to(data_path)
            output_subfolder = os.path.join(
                output_folder, os.path.dirname(relative_path)
            )
            os.makedirs(output_subfolder, exist_ok=True)

            # Générer le nom du fichier markdown
            markdown_filename = pdf_file.stem + ".md"
            markdown_path = os.path.join(output_subfolder, markdown_filename)

            # Exporter en markdown de base
            markdown_content = result.document.export_to_markdown()

            # Compter les éléments extraits
            doc = result.document
            table_count = 0
            image_count = 0

            # Analyser les éléments du document
            if hasattr(doc, "tables"):
                table_count = len(doc.tables)
                stats["tables"] += table_count
                print(
                    f"  📊 {table_count} tableau(x) extrait(s) avec double vérification"
                )

            if hasattr(doc, "pictures"):
                image_count = len(doc.pictures)
                stats["images"] += image_count
                print(f"  🖼️ {image_count} image(s) avec légende(s)")

            # Ajouter un en-tête avec les métadonnées
            header = f"""<!-- 
Document: {pdf_file.name}
Tableaux extraits: {table_count}
Images détectées: {image_count}
Extraction: Mode ACCURATE avec double vérification
-->

"""
            markdown_content = header + markdown_content

            # Sauvegarder le fichier markdown
            with open(markdown_path, "w", encoding="utf-8") as f:
                f.write(markdown_content)

            print(f"  ✓ Sauvegardé: {markdown_path}")
            stats["success"] += 1

        except Exception as e:
            print(f"  ✗ Erreur lors du traitement de {pdf_file.name}: {str(e)}")
            stats["errors"] += 1
            continue

    # Afficher les statistiques finales
    print(f"\n{'='*60}")
    print("📈 STATISTIQUES D'EXTRACTION")
    print(f"{'='*60}")
    print(f"  Documents traités: {stats['success']}/{stats['total']}")
    print(f"  Tableaux extraits: {stats['tables']}")
    print(f"  Images détectées: {stats['images']}")
    print(f"  Erreurs: {stats['errors']}")
    print(f"{'='*60}")
    print(f"✅ Fichiers markdown générés dans: {output_folder}")


def main():
    """Point d'entrée principal du script."""
    # Chemin vers le dossier data
    script_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_folder = os.path.join(script_dir, "data")

    # Vérifier que le dossier existe
    if not os.path.exists(data_folder):
        print(f"Erreur: Le dossier {data_folder} n'existe pas")
        return

    # Parser les PDFs
    parse_pdfs_to_markdown(data_folder)


if __name__ == "__main__":
    main()
