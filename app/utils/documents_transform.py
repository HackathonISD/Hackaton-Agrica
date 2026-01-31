"""
Module pour parser les fichiers PDF avec docling et générer des fichiers markdown.
"""

import os
from pathlib import Path
from docling.document_converter import DocumentConverter


def parse_pdfs_to_markdown(data_folder: str, output_folder: str = None):
    """
    Parse tous les fichiers PDF dans le dossier data et génère un fichier markdown pour chacun.

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

    # Initialiser le convertisseur docling
    converter = DocumentConverter()

    # Trouver tous les fichiers PDF
    data_path = Path(data_folder)
    pdf_files = list(data_path.rglob("*.pdf"))

    print(f"Trouvé {len(pdf_files)} fichiers PDF à traiter")

    # Parser chaque PDF
    for idx, pdf_file in enumerate(pdf_files, 1):
        try:
            print(f"\n[{idx}/{len(pdf_files)}] Traitement de: {pdf_file.name}")

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

            # Exporter en markdown
            markdown_content = result.document.export_to_markdown()

            # Sauvegarder le fichier markdown
            with open(markdown_path, "w", encoding="utf-8") as f:
                f.write(markdown_content)

            print(f"✓ Sauvegardé: {markdown_path}")

        except Exception as e:
            print(f"✗ Erreur lors du traitement de {pdf_file.name}: {str(e)}")
            continue

    print(f"\n{'='*60}")
    print(f"Traitement terminé! Fichiers markdown générés dans: {output_folder}")


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
