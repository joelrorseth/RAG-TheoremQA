import logging
import requests
from typing import List
from config import (
    REFERENCE_PDF_TEXTBOOK_FILENAME,
    REFERENCE_PDF_TEXTBOOKS,
    REFERENCE_PDFS_PATH,
    TextbookIdentifier
)


def download_pdf_textbook(textbook: TextbookIdentifier) -> None:
    """
    Download a PDF textbook, if it hasn't already been downloaded.
    """
    textbook_dir_path = REFERENCE_PDFS_PATH / textbook.name
    textbook_pdf_path = textbook_dir_path / REFERENCE_PDF_TEXTBOOK_FILENAME

    # Ensure the textbook directory exists
    textbook_dir_path.mkdir(parents=True, exist_ok=True)

    # Download the textbook PDF file if it doesn't already exist
    logging.info(f"Downloading PDF textbook '{textbook.name}'")
    if not textbook_pdf_path.exists():
        response = requests.get(textbook.source_url)
        response.raise_for_status()

        with open(textbook_pdf_path, 'wb') as f:
            f.write(response.content)
    else:
        logging.info(f"Already downloaded PDF textbook '{textbook.name}', skipping")


def download_pdf_textbooks(textbooks: List[TextbookIdentifier]) -> None:
    for textbook in textbooks:
        download_pdf_textbook(textbook)


def download_all_pdf_textbooks():
    download_pdf_textbooks(REFERENCE_PDF_TEXTBOOKS)
